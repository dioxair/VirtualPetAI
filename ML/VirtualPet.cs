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

    public VirtualPet()
    {
        SentimentDataProvider provider = new();

        qTable = new Dictionary<string, Dictionary<string, double>>
        {
            { "Happy", new Dictionary<string, double> { { "Feed", 5 }, { "Play", 4 }, { "Scold", -5 } } },
            { "Sad", new Dictionary<string, double> { { "Feed", 3 }, { "Play", 5 }, { "Scold", -2 } } },
            { "Excited", new Dictionary<string, double> { { "Feed", 4 }, { "Play", 6 }, { "Scold", -7 } } },
            { "Grumpy", new Dictionary<string, double> { { "Feed", 2 }, { "Play", -2 }, { "Scold", -6 } } },
            { "Neutral", new Dictionary<string, double> { { "Feed", 3 }, { "Play", 3 }, { "Scold", -3 } } },
            { "Tired", new Dictionary<string, double> { { "Feed", 2 }, { "Play", -1 }, { "Scold", -3 } } }
        };
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
        var dataView = mlContext.Data.LoadFromEnumerable(provider.SampleData);
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        Console.WriteLine("Training model...");
        var model = pipeline.Fit(dataView);
        sentimentEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        Console.Clear();

#if DEBUG
        SaveModel(mlContext, model, dataView, "PetSentimentAnalysisModel.zip");
#endif
    }

#if DEBUG
    public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
    {
        DataViewSchema dataViewSchema = data.Schema;

        using (var fs = File.Create(modelSavePath))
        {
            mlContext.Model.Save(model, dataViewSchema, fs);
        }
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
        reward *= Math.Max(0.5, 1.0 + (emotionScore / 50.0));

        emotionScore += (int)reward;
        UpdateState();

        double maxFutureReward = GetMaxFutureReward(state);
        qTable[state][action] = (1 - Alpha) * qTable[state][action] + Alpha * (reward + Gamma * maxFutureReward);

        Console.WriteLine($"Pet reacts to {action}: Reward {reward}");
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

        if (lastIgnoredAction != null)
        {
            ignoredActionsQTable[lastIgnoredAction] = (1 - Alpha) * ignoredActionsQTable[lastIgnoredAction] + Alpha * 2;
            Console.WriteLine(
                $"Your pet’s ignored action '{lastIgnoredAction}' successfully engaged you! Rewarding...");
            lastIgnoredAction = null;
        }

        ignoreCounter = 0;
    }

    private void UpdateState()
    {
        if (emotionScore >= 15) state = "Excited";
        else if (emotionScore <= -10) state = "Grumpy";
        else if (emotionScore >= 5) state = "Happy";
        else if (emotionScore <= -5) state = "Sad";
        else if (emotionScore < -15) state = "Tired";
        else state = "Neutral";
    }

    private double GetMaxFutureReward(string newState)
    {
        double maxReward = double.MinValue;
        foreach (var reward in qTable[newState].Values)
            if (reward > maxReward)
                maxReward = reward;
        return maxReward == double.MinValue ? 0 : maxReward;
    }

    public void ChatWithPet(string userInput)
    {
        if (userInput == "")
        {
            CheckIfIgnored();
            return;
        }

        var prediction = sentimentEngine.Predict(new SentimentData { Text = userInput });
        string sentiment = prediction.Prediction ? "Positive" : "Negative";
        Console.WriteLine(
            $"Pet detects sentiment: {sentiment} | Positive score: {Math.Round(prediction.Probability * 100)}%");

        if (sentiment == "Positive")
            emotionScore += 3;
        else
            emotionScore -= 3;

        if (lastIgnoredAction != null)
        {
            double reward = sentiment == "Positive" ? 2 : -1;
            ignoredActionsQTable[lastIgnoredAction] =
                (1 - Alpha) * ignoredActionsQTable[lastIgnoredAction] + Alpha * reward;

            Console.WriteLine($"Your pet’s ignored action '{lastIgnoredAction}' received a reward of {reward}.");
            lastIgnoredAction = null;
        }

        UpdateState();
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

        ignoreCounter = 0;
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