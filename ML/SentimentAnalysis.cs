using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace VirtualPetAI.ML
{
    public class SentimentAnalysis
    {
        private readonly MLContext mlContext;
        public PredictionEngine<SentimentData, SentimentPrediction> SentimentEngine { get; }

        public SentimentAnalysis()
        {
            mlContext = new MLContext();
            SentimentEngine = TrainSentimentModel();
        }

        private PredictionEngine<SentimentData, SentimentPrediction> TrainSentimentModel()
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
                "ML/Training/TrainingData.tsv",
                separatorChar: '\t', 
                hasHeader: true);

            EstimatorChain<KeyToValueMappingTransformer> pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training model...");
            TransformerChain<KeyToValueMappingTransformer> model = pipeline.Fit(dataView);
            Console.Clear();

#if DEBUG
            SaveModel(model, dataView, "PetSentimentAnalysisModel.zip");
#endif

            return mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        }

#if DEBUG
        private void SaveModel(ITransformer model, IDataView data, string modelSavePath)
        {
            using FileStream fs = File.Create(modelSavePath);
            mlContext.Model.Save(model, data.Schema, fs);
        }
#endif
    }

    public class SentimentData
    {
        [LoadColumn(0)] public string Text { get; set; }
        [LoadColumn(1)] public string Label { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public string Prediction { get; set; }
        public float[] Score { get; set; }
    }

}
