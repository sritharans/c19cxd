//////////////////////////////////////////////////////////////////////////////////////////////////////////
// About: This is a program to train and test a DNN for image classification using the
//        transfer learning method. This program could be customized to utlize any of
//        the supported pretrained model, such as InceptionV3 or ResNetV2, with tiny changes.
//        This program has been adapted to use the COVID-19 dataset from here:
//        https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
//        The dataset contains X-ray images of patients that are categorized as:
//          - Typically healthy (NORMAL)
//          - Confirmed COVID-19 cases (COVID-19)
//          - Other viral pneumonia cases (PNEUMONIA)
//        This program trains and evaluates the selected model, to identify where the patient is
//        classified according to the above categories.
//
// Author: Sritharan Sivaguru (WQD180086/17198431)
//
// Learn more about F# at http://fsharp.org

// System runtime libraries
open System
open System.IO

// ML.Net runtime libraries for Image classification
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Vision

// This is the input schema that will be used by image loader to capture the orignal classification
// and image data path. This will make up our initial data set.
[<CLIMutable>]
type ImageData =
    { ImagePath: string // Path to the image to be loaded for training/validation/testing
      Label: string } // The original classification label for the image (COVID-19/NORMAL/PNEUMONIA)

// This schema will be used by the prediction model to predict the classification of the image.
// The predicted classification could then be compared with the original classification.
[<CLIMutable>]
type ImagePrediction =
    { ImagePath: string // Path to the image to be loaded for prediction
      Label: string // The original classification label for the image (COVID-19/NORMAL/PNEUMONIA)
      PredictedLabel: string } // The predicted classification label by the Image classification model

// To time an execution of a function
let duration f =
    let timer = Diagnostics.Stopwatch()
    timer.Start()
    let returnValue = f ()
    timer.Stop()
    printfn "Elapsed Time: %f seconds" timer.Elapsed.TotalSeconds
    returnValue

// Takes the location/path of the image directory and recursively loads their location into ImageData.
// Returns an array containing the the image location (ImagePath) and it's classification (label)
// Only the root folder/directory needs to be provided in the 'path' argument
let LoadImagesDataset (path: string) =
    // Get all files in all subdirectories
    Directory.GetFiles(path, "*", SearchOption.AllDirectories)
    // Filter out files that are not in PNG format
    |> Array.filter (fun file -> (Path.GetExtension(file) = ".png"))
    // Get the full path for the image file and the directory it was stored in as the Label
    |> Array.Parallel.map
        (fun file ->
            { ImagePath = file
              Label = Directory.GetParent(file).Name })

// Main program body
[<EntryPoint>]
let main argv =

    match argv.Length with
    | 2 ->
        printfn ">---------->"

        printfn "[ Model: %A | Dataset: %s ]"
        <| enum<ImageClassificationTrainer.Architecture> (int argv.[0])
        <| argv.[1]

        printfn "<----------<"
    | _ ->
        printfn "Usage: dotnet run <0/1/2/3> <dataset directory>"
        printfn "0 - ResnetV2-101\n1 - InceptionV3\n2 - MobilenetV2\n3 - ResnetV2-50"
        exit (0)

    // Get the current project directory and path to the X-ray images
    // The COVID-19_Radiography_Database folder contains X-ray images of patients that are:
    // - Typically healthy (NORMAL)
    // - Confirmed COVID-19 cases (COVID-19)
    // - Other viral pneumonia cases (PNEUMONIA)
    let datasetDir = argv.[1]

    // First we load the images into our ImageData schema
    let images = LoadImagesDataset datasetDir

    // We then initialize the MLContext. It provides a way for us to create components for
    // data preparation, feature enginering, training, prediction and model evaluation.
    let mlContext = MLContext(0)

    // We then load the images into an IDataView type (it is analogous to a DataFrame in Pandas)
    // and shuffle the rows in the IDataView so it'll be better balanced
    let data =
        images
        |> mlContext.Data.LoadFromEnumerable
        |> mlContext.Data.ShuffleRows

    // Output some info about our Dataset, like the total number of images
    printfn "[ Total images: %d ]" (int64 (data.GetRowCount()))
    // Display the first 5 rows in our Dataset
    printfn ">---------->"

    printfn
        "Dataset Sample: %A"
        (mlContext.Data.CreateEnumerable<ImageData>(data, true)
         |> Seq.take 5)

    printfn "<----------<"

    // Create our data pipeline that will be used by our model for training and testing
    // This data pipeline will also contain the raw image data from each ImagePath
    // Each Label is also converted to a value key (0 - COVID-19, 1 - NORMAL, 2 - PNEUMONIA)
    let dataPipeline =
        EstimatorChain()
            .Append(mlContext.Transforms.Conversion.MapValueToKey("LabelAsKey", "Label"))
            .Append(mlContext.Transforms.LoadRawImageBytes("Image", null, "ImagePath"))

    // We fit and transform the data to prepare it for training and testing
    let preppedData =
        let processingTransformer = data |> dataPipeline.Fit
        data |> processingTransformer.Transform

    // We now split the data into train, validation and test sets
    // The first 70% of the data is used to train the model.
    // Out of the remaining 30%, 90% is used for validation and the remaining 10% is used for testing
    let train, validate, test =
        preppedData
        |> (fun originalData ->
            let trainValSplit =
                mlContext.Data.TrainTestSplit(originalData, 0.7)

            let testValSplit =
                mlContext.Data.TrainTestSplit(trainValSplit.TestSet)

            (trainValSplit.TrainSet, testValSplit.TrainSet, testValSplit.TestSet))

    // Here we define the options for the ImageClassificationTrainer DNN
    // Currently the classifier supports the following DNN architerctures:
    // - ResnetV2-101
    // - InceptionV3
    // - MobilenetV2
    // - ResnetV2-50
    let dnnOptions = ImageClassificationTrainer.Options()
    dnnOptions.FeatureColumnName <- "Image"
    dnnOptions.LabelColumnName <- "LabelAsKey"
    dnnOptions.ValidationSet <- validate
    dnnOptions.Arch <- enum<ImageClassificationTrainer.Architecture> (int argv.[0])

    dnnOptions.MetricsCallback <-
        Action<ImageClassificationTrainer.ImageClassificationMetrics>(fun x -> printfn "[ %s ]" (x.ToString()))

    // We define the inputs for the training model pipeline
    let trainingPipeline =
        EstimatorChain()
            .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(dnnOptions))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "LabelAsKey"))

    // We then train and time the model
    printfn ">---------->"
    printfn "Training the model: %A" dnnOptions.Arch

    let trainedModel =
        duration (fun () -> train |> trainingPipeline.Fit)

    printfn "<----------<"

    // We now define the prediction transform for the test data set
    let predictions = test |> trainedModel.Transform

    // We then evaluate the prediction model
    printfn ">---------->"
    printfn "Evaluating model: %A" dnnOptions.Arch

    let metrics =
        duration (fun () -> mlContext.MulticlassClassification.Evaluate(predictions, "LabelAsKey"))

    printfn "<----------<"
    printfn "[ MacroAccurracy: %f | LogLoss: %f ]" metrics.MacroAccuracy metrics.LogLoss

    // We save our model so that we can reload it later if needed
    mlContext.Model.Save(
        trainedModel,
        preppedData.Schema,
        Path.Join(Directory.GetCurrentDirectory(), "model_" + argv.[0] + ".zip")
    )

    // Display the first 50 predictions from the model evaluation
    printfn ">---------->"
    printfn "Display the first 50 predictions:"

    mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true)
    |> Seq.take 50
    |> Seq.iter (fun p -> printfn "[ Original: %s | Predicted: %s | Image: %s ]" p.Label p.PredictedLabel p.ImagePath)

    printfn "<----------<"

    // Exit the program
    0 // return an integer exit code
