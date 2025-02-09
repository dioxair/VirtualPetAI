using Microsoft.ML;

namespace VirtualPetAI.ML;

internal class VirtualPet
{
    private const double Alpha = 0.1; // Learning rate
    private const double Gamma = 0.9; // Discount factor
    private const int ignoreThreshold = 3;
    private int emotionScore; // Tracks long-term emotional trends

    private int ignoreCounter;

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
        // Pull the data schema from the IDataView used for training the model
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

        // Update emotion score based on action
        emotionScore += (int)reward;

        UpdateState();

        // Q(s, a) = (1 - Alpha) * Q(s, a) + Alpha * (reward + Gamma * maxFutureReward)
        //
        // Explanation:
        // - Q(s, a) represents the expected future reward for taking action 'a' in state 's'.
        // - Alpha (α) is the learning rate (0 < α ≤ 1), which controls how much new information overrides old Q-values.
        //   - If α is close to 0, learning is slow (old experiences dominate).
        //   - If α is close to 1, learning is fast (new experiences dominate).
        // - reward is the immediate reward received after taking action 'a' in state 's'.
        // - Gamma (γ) is the discount factor (0 ≤ γ ≤ 1), which determines how much future rewards matter:
        //   - If γ = 0, the agent only considers immediate rewards.
        //   - If γ is close to 1, the agent values future rewards more.
        // - maxFutureReward is the highest Q-value of the next possible state (i.e., best possible future reward).
        //
        // Breakdown of each term:
        // 1. (1 - Alpha) * Q(s, a) -> Keeps some of the old Q-value to prevent drastic changes.
        // 2. reward -> Adds the immediate reward for taking action 'a'.
        // 3. Gamma * maxFutureReward -> Encourages the agent to take actions that lead to better long-term rewards.
        //
        // The formula **blends old knowledge with new experiences**, slowly refining the Q-values to optimize decision-making.
        double maxFutureReward = GetMaxFutureReward(state);
        qTable[state][action] = (1 - Alpha) * qTable[state][action] + Alpha * (reward + Gamma * maxFutureReward);

        Console.WriteLine($"Pet reacts to {action}: Reward {reward}");
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

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

        UpdateState();
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");

        ignoreCounter = 0;
    }

    public void CheckIfIgnored()
    {
        ignoreCounter++;

        if (ignoreCounter >= ignoreThreshold)
        {
            string chosenAction;

            string[] possibleActionsSad = ["Whimper", "Hide", "Avoid you", "Sleep"];
            string[] possibleActionsHappy = ["Play", "Make noise", "Jump", "Spin around"];
            string[] possibleActionsNeutral = ["Make noise", "Look at you", "Wag tail"];

            if (emotionScore <= -5)
                chosenAction = possibleActionsSad[random.Next(possibleActionsSad.Length)];
            else if (emotionScore >= 5)
                chosenAction = possibleActionsHappy[random.Next(possibleActionsHappy.Length)];
            else
                chosenAction = possibleActionsNeutral[random.Next(possibleActionsNeutral.Length)];

            emotionScore -= 3;
            Console.WriteLine($"Your pet is feeling ignored! It decides to {chosenAction}.");
            Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");
            ignoreCounter = 0;
        }
    }
}