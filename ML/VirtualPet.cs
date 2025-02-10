using Microsoft.ML;

namespace VirtualPetAI.ML;

internal class VirtualPet
{
    private const double Alpha = 0.1; // Learning rate
    private const double Epsilon = 0.2; // Exploration rate
    private const double Gamma = 0.9; // Discount factor
    private const int ignoreThreshold = 3;
    private int emotionScore; // Tracks long-term emotional trends

    private int ignoreCounter;
    private readonly Dictionary<string, double> ignoredActionsQTable;
    private string lastIgnoredAction; // Store the last ignored-response action

    private readonly MLContext mlContext;
    private readonly Dictionary<string, Dictionary<string, double>> qTable;
    private readonly Random random = new();
    private readonly PredictionEngine<SentimentData, SentimentPrediction> sentimentEngine;
    private string state;

    private readonly HashSet<string> positiveActions = ["Feed", "Play", "Pet", "Praise"];
    private readonly HashSet<string> negativeActions = ["Scold", "Yell", "Take away toy"];

    public VirtualPet()
    {
        qTable = new Dictionary<string, Dictionary<string, double>> {
        { "Happy", new() { { "Feed", 5 }, { "Play", 4 }, { "Pet", 6 }, { "Praise", 5 }, { "Scold", -5 }, { "Yell", -4 }, { "Take away toy", -6 } } },
        { "Sad", new() { { "Feed", 3 }, { "Play", 5 }, { "Pet", 4 }, { "Praise", 3 }, { "Scold", -2 }, { "Yell", -5 }, { "Take away toy", -4 } } },
        { "Excited", new() { { "Feed", 4 }, { "Play", 6 }, { "Pet", 7 }, { "Praise", 6 }, { "Scold", -7 }, { "Yell", -5 }, { "Take away toy", -8 } } },
        { "Grumpy", new() { { "Feed", 2 }, { "Play", -2 }, { "Pet", 3 }, { "Praise", 2 }, { "Scold", -6 }, { "Yell", -4 }, { "Take away toy", -5 } } },
        { "Neutral", new() { { "Feed", 3 }, { "Play", 3 }, { "Pet", 4 }, { "Praise", 3 }, { "Scold", -3 }, { "Yell", -3 }, { "Take away toy", -3 } } },
        { "Tired", new() { { "Feed", 2 }, { "Play", -1 }, { "Pet", 3 }, { "Praise", 2 }, { "Scold", -3 }, { "Yell", -4 }, { "Take away toy", -5 } } } };
        state = "Neutral";

        ignoredActionsQTable = new Dictionary<string, double>
        {
            { "Whimper", 0 },
            { "Hide", 0 },
            { "Avoid you", 0 },
            { "Sleep", 0 },
            { "Play", 0 },
            { "Make noise", 0 },
            { "Jump", 0 },
            { "Spin around", 0 },
            { "Look at you", 0 },
            { "Wag tail", 0 }
        };

        mlContext = new MLContext();
        IDataView dataView =
            mlContext.Data.LoadFromTextFile<SentimentData>("training_large.tsv", separatorChar: '\t', hasHeader: true);

        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        Console.WriteLine("Training model...");
        var model = pipeline.Fit(dataView);
        sentimentEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        Console.Clear();

#if DEBUG
        SaveModel(mlContext, model, dataView, "PetSentimentAnalysisModel.zip");
#endif
    }

#if DEBUG
    private static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
    {
        DataViewSchema dataViewSchema = data.Schema;

        using FileStream fs = File.Create(modelSavePath);
        mlContext.Model.Save(model, dataViewSchema, fs);
    }
#endif

    public void PerformAction(string action)
    {
        if (!qTable[state].ContainsKey(action))
        {
            Console.WriteLine("Invalid action.");
            return;
        }

        double reward = qTable[state][action];

        if (positiveActions.Contains(action))
        {
            reward = Math.Abs(reward); // Ensure positive reward
        }
        else if (negativeActions.Contains(action))
        {
            reward = -Math.Abs(reward); // Ensure negative reward
        }

        // Modify reward scaling based on emotional state
        reward *= Math.Max(0.5, 1.0 + (emotionScore / 50.0));

        emotionScore += (int)reward;
        UpdateState();

        double maxFutureReward = GetMaxFutureReward(state);
        qTable[state][action] = (1 - Alpha) * qTable[state][action] + Alpha * (reward + Gamma * maxFutureReward);

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
        if (emotionScore >= 15) state = "Excited";
        else if (emotionScore >= 5) state = "Happy";
        else if (emotionScore <= -15) state = "Tired";
        else if (emotionScore <= -10) state = "Grumpy";
        else if (emotionScore <= -5) state = "Sad";
        else state = "Neutral";
    }

    private double GetMaxFutureReward(string newState)
    {
        double maxReward = qTable[newState].Values.Prepend(double.MinValue).Max();
        return maxReward == double.MinValue ? 0 : maxReward;
    }

    public void ChatWithPet(string userInput)
    {
        if (string.IsNullOrEmpty(userInput) || userInput == "Ignore")
        {
            CheckIfIgnored();
            return;
        }

        SentimentPrediction prediction = sentimentEngine.Predict(new SentimentData { Text = userInput });
        Console.WriteLine($"Pet detects sentiment: {prediction.Prediction}");
        Console.WriteLine(
            $"Positive score: {Math.Round(prediction.Score[0] * 100, 1)}% | Neutral score: {Math.Round(prediction.Score[1] * 100, 1)}% | Negative score: {Math.Round(prediction.Score[2] * 100, 1)}%");

        switch (prediction.Prediction)
        {
            case "Positive":
                emotionScore += 3;
                break;
            case "Neutral":
                emotionScore += 0;
                break;
            case "Negative":
                emotionScore -= 3;
                break;
        }

        UpdateState();
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");
    }

    public void CheckIfIgnored()
    {
        ignoreCounter++;

        if (ignoreCounter >= ignoreThreshold)
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