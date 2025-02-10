using Microsoft.ML;

namespace VirtualPetAI.ML;

internal class VirtualPet
{
    private const double Alpha = 0.1; // Learning rate
    private const double Epsilon = 0.2; // Exploration rate
    private const double Gamma = 0.9; // Discount factor
    private const int IgnoreThreshold = 3;

    private int emotionScore = 0;
    private int ignoreCounter = 0;
    private string? lastIgnoredAction;
    private string state = "Neutral";

    private readonly Dictionary<string, Dictionary<string, double>> qTable;
    private readonly Dictionary<string, double> ignoredActionsQTable;
    private readonly HashSet<string> positiveActions = ["Feed", "Play", "Pet", "Praise"];
    private readonly HashSet<string> negativeActions = ["Scold", "Yell", "Take away toy"];
    private readonly Random random = new();
    private readonly PredictionEngine<SentimentData, SentimentPrediction> sentimentEngine;

    public VirtualPet()
    {
        qTable = InitializeQTable();
        ignoredActionsQTable = InitializeIgnoredActionsQTable();
        sentimentEngine = new SentimentAnalysis().SentimentEngine;
    }

    private static Dictionary<string, Dictionary<string, double>> InitializeQTable() => new()
    {
        ["Happy"] = new Dictionary<string, double> { ["Feed"] = 5, ["Play"] = 4, ["Pet"] = 6, ["Praise"] = 5, ["Scold"] = -5, ["Yell"] = -4, ["Take away toy"] = -6 },
        ["Sad"] = new Dictionary<string, double> { ["Feed"] = 3, ["Play"] = 5, ["Pet"] = 4, ["Praise"] = 3, ["Scold"] = -2, ["Yell"] = -5, ["Take away toy"] = -4 },
        ["Excited"] = new Dictionary<string, double> { ["Feed"] = 4, ["Play"] = 6, ["Pet"] = 7, ["Praise"] = 6, ["Scold"] = -7, ["Yell"] = -5, ["Take away toy"] = -8 },
        ["Grumpy"] = new Dictionary<string, double> { ["Feed"] = 2, ["Play"] = -2, ["Pet"] = 3, ["Praise"] = 2, ["Scold"] = -6, ["Yell"] = -4, ["Take away toy"] = -5 },
        ["Neutral"] = new Dictionary<string, double> { ["Feed"] = 3, ["Play"] = 3, ["Pet"] = 4, ["Praise"] = 3, ["Scold"] = -3, ["Yell"] = -3, ["Take away toy"] = -3 },
        ["Tired"] = new Dictionary<string, double> { ["Feed"] = 2, ["Play"] = -1, ["Pet"] = 3, ["Praise"] = 2, ["Scold"] = -3, ["Yell"] = -4, ["Take away toy"] = -5 }
    };

    private static Dictionary<string, double> InitializeIgnoredActionsQTable() => new()
    {
        ["Whimper"] = 0,
        ["Hide"] = 0,
        ["Avoid you"] = 0,
        ["Sleep"] = 0,
        ["Play"] = 0,
        ["Make noise"] = 0,
        ["Jump"] = 0,
        ["Spin around"] = 0,
        ["Look at you"] = 0,
        ["Wag tail"] = 0
    };

    public void PerformAction(string action)
    {
        if (!qTable[state].TryGetValue(action, out double reward))
        {
            Console.WriteLine("Invalid action.");
            return;
        }

        reward = positiveActions.Contains(action) ? Math.Abs(reward) : -Math.Abs(reward);
        reward *= Math.Max(0.5, 1.0 + (emotionScore / 50.0));

        emotionScore += (int)reward;
        UpdateState();
        qTable[state][action] = (1 - Alpha) * qTable[state][action] + Alpha * (reward + Gamma * GetMaxFutureReward());

        Console.WriteLine($"Pet reacts to {action}: Reward {reward}");
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

        if (lastIgnoredAction != null)
        {
            double ignoredReward = positiveActions.Contains(action) ? 2 : -2;
            ignoredActionsQTable[lastIgnoredAction] =
                (1 - Alpha) * ignoredActionsQTable[lastIgnoredAction] + Alpha * ignoredReward;

            Console.WriteLine(
                $"Your pet's ignored action '{lastIgnoredAction}' {(ignoredReward > 0 ? "positively engaged you! Rewarding..." : "was negative. Penalizing...")}");
            Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

            lastIgnoredAction = null;
        }
        ignoreCounter = 0;
    }

    private void UpdateState()
    {
        state = emotionScore switch
        {
            >= 15 => "Excited",
            >= 5 => "Happy",
            <= -15 => "Tired",
            <= -10 => "Grumpy",
            <= -5 => "Sad",
            _ => "Neutral"
        };
    }

    private double GetMaxFutureReward() => qTable[state].Values.DefaultIfEmpty(0).Max();

    public void ChatWithPet(string userInput)
    {
        if (string.IsNullOrEmpty(userInput) || userInput == "Ignore")
        {
            CheckIfIgnored();
            return;
        }

        SentimentPrediction prediction = sentimentEngine.Predict(new SentimentData { Text = userInput });
        emotionScore += prediction.Prediction switch { "Positive" => 3, "Negative" => -3, _ => 0 };

        Console.WriteLine($"Pet detects sentiment: {prediction.Prediction}");
        Console.WriteLine(
            $"Positive score: {Math.Round(prediction.Score[0] * 100, 1)}% | Neutral score: {Math.Round(prediction.Score[1] * 100, 1)}% | Negative score: {Math.Round(prediction.Score[2] * 100, 1)}%");

        UpdateState();
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");
    }

    public void CheckIfIgnored()
    {
        ignoreCounter++;

        if (ignoreCounter >= IgnoreThreshold)
        {
            // Epsilon-Greedy Algorithm
            string chosenAction;
            if (random.NextDouble() < Epsilon)
                chosenAction = ignoredActionsQTable.Keys.ElementAt(random.Next(ignoredActionsQTable.Count));
            else
                chosenAction = ignoredActionsQTable.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;

            lastIgnoredAction = chosenAction; // Store for potential reward

            emotionScore -= 3; // Pet feels ignored
            UpdateState();
            Console.WriteLine($"Your pet is feeling ignored! It decides to {chosenAction}.");
            Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");
            ignoreCounter = 0;
        }
    }
}