using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VirtualPetAI.ML
{
    public class SentimentDataProvider
    {
        // Instance property
        public List<SentimentData> SampleData { get; } =
        [
            new SentimentData { Text = "I love you, buddy!", Label = true },
            new SentimentData { Text = "You're the best pet ever!", Label = true },
            new SentimentData { Text = "Such a cute and smart little creature!", Label = true },
            new SentimentData { Text = "You're my favorite pet!", Label = true },
            new SentimentData { Text = "I enjoy spending time with you!", Label = true },
            new SentimentData { Text = "Good job! You're such a clever pet!", Label = true },
            new SentimentData { Text = "You're so adorable!", Label = true },
            new SentimentData { Text = "I love playing with you!", Label = true },

            // Neutral Sentiments
            new SentimentData { Text = "Are you hungry?", Label = true }, // Slightly positive
            new SentimentData { Text = "What do you want to do today?", Label = true },
            new SentimentData { Text = "It's time for your walk.", Label = true },
            new SentimentData { Text = "How are you feeling?", Label = true },
            new SentimentData { Text = "I'm home!", Label = true },
            new SentimentData { Text = "Do you need anything?", Label = true },
            new SentimentData { Text = "Let's go outside.", Label = true },

            // Negative Sentiments
            new SentimentData { Text = "You're being really annoying.", Label = false },
            new SentimentData { Text = "Stop that right now!", Label = false },
            new SentimentData { Text = "Bad pet! Don't do that!", Label = false },
            new SentimentData { Text = "I don't like you right now.", Label = false },
            new SentimentData { Text = "You never listen to me.", Label = false },
            new SentimentData { Text = "I'm getting tired of this.", Label = false },
            new SentimentData { Text = "You're not behaving well today.", Label = false },
            new SentimentData { Text = "Ugh, you're so frustrating!", Label = false }
        ];
    }

    public class SentimentData
    {
        [LoadColumn(0)] public string Text { get; set; }
        [LoadColumn(1)] public bool Label { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
