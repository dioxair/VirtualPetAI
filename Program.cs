using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

class VirtualPet
{
    private Dictionary<string, Dictionary<string, double>> qTable;
    private string state;
    private Random random;
    private const double Alpha = 0.1;  // Learning rate
    private const double Gamma = 0.9;  // Discount factor
    private const double Epsilon = 0.2; // Exploration rate
    private int emotionScore = 0; // Tracks long-term emotional trends

    private MLContext mlContext;
    private PredictionEngine<SentimentData, SentimentPrediction> sentimentEngine;

    public VirtualPet()
    {
        qTable = new Dictionary<string, Dictionary<string, double>>()
        {
            { "Happy", new Dictionary<string, double>{{"Feed", 5}, {"Play", 4}, {"Scold", -5}} },
            { "Sad", new Dictionary<string, double>{{"Feed", 3}, {"Play", 5}, {"Scold", -2}} },
            { "Excited", new Dictionary<string, double>{{"Feed", 4}, {"Play", 6}, {"Scold", -7}} },
            { "Grumpy", new Dictionary<string, double>{{"Feed", 2}, {"Play", -2}, {"Scold", -6}} },
            { "Neutral", new Dictionary<string, double>{{"Feed", 3}, {"Play", 3}, {"Scold", -3}} },
            { "Tired", new Dictionary<string, double>{{"Feed", 2}, {"Play", -1}, {"Scold", -3}} }
        };
        state = "Neutral";
        random = new Random();

        // I need to get more sample data.
        var sampleData = new List<SentimentData>
        {
            // Positive Sentiments
            new() { Text = "I love you, buddy!", Label = true },
            new() { Text = "You're the best pet ever!", Label = true },
            new() { Text = "Such a cute and smart little creature!", Label = true },
            new() { Text = "You're my favorite pet!", Label = true },
            new() { Text = "I enjoy spending time with you!", Label = true },
            new() { Text = "Good job! You're such a clever pet!", Label = true },
            new() { Text = "You're so adorable!", Label = true },
            new() { Text = "I love playing with you!", Label = true },

            // Neutral Sentiments
            new() { Text = "Are you hungry?", Label = true }, // Slightly positive
            new() { Text = "What do you want to do today?", Label = true },
            new() { Text = "It's time for your walk.", Label = true },
            new() { Text = "How are you feeling?", Label = true },
            new() { Text = "I'm home!", Label = true },
            new() { Text = "Do you need anything?", Label = true },
            new() { Text = "Let's go outside.", Label = true },

            // Negative Sentiments
            new() { Text = "You're being really annoying.", Label = false },
            new() { Text = "Stop that right now!", Label = false },
            new() { Text = "Bad pet! Don't do that!", Label = false },
            new() { Text = "I don't like you right now.", Label = false },
            new() { Text = "You never listen to me.", Label = false },
            new() { Text = "I'm getting tired of this.", Label = false },
            new() { Text = "You're not behaving well today.", Label = false },
            new() { Text = "Ugh, you're so frustrating!", Label = false },
        };

        mlContext = new MLContext();
        var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        Console.WriteLine("Training model...");
        var model = pipeline.Fit(dataView);
        sentimentEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        Console.Clear();
    }

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

        // Q-learning update rule:
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
        {
            if (reward > maxReward)
                maxReward = reward;
        }
        return maxReward == double.MinValue ? 0 : maxReward;
    }

    public void ChatWithPet(string userInput)
    {
        var prediction = sentimentEngine.Predict(new SentimentData { Text = userInput });
        string sentiment = prediction.Prediction ? "Positive" : "Negative";
        Console.WriteLine($"Pet detects sentiment: {sentiment} | Positive score: {Math.Round(prediction.Probability * 100)}%");

        if (sentiment == "Positive")
            emotionScore += 3;
        else
            emotionScore -= 3;

        UpdateState();
        Console.WriteLine($"Pet state: {state} | Emotion Score: {emotionScore}");
    }

    public void LearnFromExperience()
    {
        string action = ChooseAction(state);
        Console.WriteLine($"Pet decides to: {action}");
        PerformAction(action);
    }

    private string ChooseAction(string currentState)
    {
        if (random.NextDouble() < Epsilon)
        {
            var actions = new List<string>(qTable[currentState].Keys);
            return actions[random.Next(actions.Count)];
        }
        else
        {
            string bestAction = null;
            double maxReward = double.MinValue;
            foreach (var action in qTable[currentState])
            {
                if (action.Value > maxReward)
                {
                    maxReward = action.Value;
                    bestAction = action.Key;
                }
            }
            return bestAction;
        }
    }
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

class Program
{
    static void Main(string[] args)
    {
        VirtualPet pet = new();
        Console.WriteLine("Interact with your virtual AI pet! Type \"Feed\", \"Play\", \"Scold\", \"Learn\" or chat with it. Type \"exit\" to exit the program.");

        while (true)
        {
            string input = Console.ReadLine();
            if (input == "exit") break;
            if (input == "Feed" || input == "Play" || input == "Scold")
                pet.PerformAction(input);
            else if (input == "Learn")
                pet.LearnFromExperience();
            else
                pet.ChatWithPet(input);
        }
    }
}
