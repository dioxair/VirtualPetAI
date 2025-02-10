using Microsoft.ML.Data;

namespace VirtualPetAI.ML
{
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
