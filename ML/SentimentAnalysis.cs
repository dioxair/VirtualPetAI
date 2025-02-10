using Microsoft.ML.Data;

namespace VirtualPetAI.ML
{
    public class SentimentDataProvider
    {
        public List<SentimentData> SampleData { get; } =
        [
            // Positive Sentiments
            new SentimentData { Text = "I love you, buddy!", Label = "Positive" },
            new SentimentData { Text = "You're the best pet ever!", Label = "Positive" },
            new SentimentData { Text = "Such a cute and smart little creature!", Label = "Positive" },
            new SentimentData { Text = "You're my favorite pet!", Label = "Positive" },
            new SentimentData { Text = "I enjoy spending time with you!", Label = "Positive" },
            new SentimentData { Text = "Good job! You're such a clever pet!", Label = "Positive" },
            new SentimentData { Text = "You're so adorable!", Label = "Positive" },
            new SentimentData { Text = "I love playing with you!", Label = "Positive" },

            // Neutral Sentiments
            new SentimentData { Text = "Are you hungry?", Label = "Neutral" },
            new SentimentData { Text = "What do you want to do today?", Label = "Neutral" },
            new SentimentData { Text = "It's time for your walk.", Label = "Neutral" },
            new SentimentData { Text = "How are you feeling?", Label = "Neutral" },
            new SentimentData { Text = "I'm home!", Label = "Neutral" },
            new SentimentData { Text = "Do you need anything?", Label = "Neutral" },
            new SentimentData { Text = "Let's go outside.", Label = "Neutral" },

            // Negative Sentiments
            new SentimentData { Text = "You're being really annoying.", Label = "Negative" },
            new SentimentData { Text = "Stop that right now!", Label = "Negative" },
            new SentimentData { Text = "Bad pet! Don't do that!", Label = "Negative" },
            new SentimentData { Text = "I don't like you right now.", Label = "Negative" },
            new SentimentData { Text = "You never listen to me.", Label = "Negative" },
            new SentimentData { Text = "I'm getting tired of this.", Label = "Negative" },
            new SentimentData { Text = "You're not behaving well today.", Label = "Negative" },
            new SentimentData { Text = "Ugh, you're so frustrating!", Label = "Negative" }
        ];
    }

    public class SentimentData
    {
        [LoadColumn(0)] public string Text { get; set; }
        [LoadColumn(1)] public string Label { get; set; } // Change to string for multi-class classification
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public string Prediction { get; set; } // Output a string category
        public float[] Score { get; set; }
    }

}
