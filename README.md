# VirtualPetAI
Pet AI using Q-Learning and Sentiment Analysis in ML.NET 

## Features
- **(Modified) Reinforcement Learning (RL) with [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)**: The pet AI learns over time by maximizing future rewards based on user interactions. However, it's not really true Q-Learning because the user still decides the actions. The pet AI learns which actions get the best approval of the user.
- **[Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)**: The pet AI can interpret the tone of the text you type and changes its state based on it.
- **State System**: The pet AI has different states like "Happy", "Sad", "Excited", etc., based on the `emotionScore`, which is affected by actions taken by the user towards the pet AI

## How it works

### 1. Q-Learning
The Q-learning formula I'm using is:

$Q(s, a) = (1 - α) * Q(s, a) + α * (reward + γ * maxFutureReward)$

Where:

- $Q(s, a)$ represents the expected future reward for taking action a in state s.
- α (Alpha) is the learning rate, controlling how much new information influences the Q-value (0 < α ≤ 1).
  - If α is close to 0, the pet AI learns slowly, relying more on past experiences for behavior
  - If α is close to 1, the pet AI learns quickly, relying more on new experiences for behavior
- reward is the immediate reward for taking action a in state s.
- γ (Gamma) is the discount factor, determining how much future rewards matter.
  - If γ = 0, only immediate rewards are considered.
  - If γ is close to 1, future rewards are given more importance.
- maxFutureReward is the best possible reward achievable from the next state.

To be honest, this took the most amount of time for me to wrap my head around since I'm not the best at math, and I still don't fully understand it.

This formula is primarily what decides how the pet AI will react to the user's actions (Feed, Praise, Scold, etc.)

### 2. Sentiment Analysis
In this project, I'm using [ML.NET](https://dotnet.microsoft.com/en-us/apps/ai/ml-dotnet) an open-source machine learning framework.

For my sentiment analysis model, I decided to use text classification, where the model receives an input text (ex: "Bad boy! Stop chewing that!") and is asked to guess the output ("Positive, "Neutral", "Negative").

In order to predict the sentiment of text, I had to somehow find a *ton* of labeled data to feed the model. I couldn't find much of anything for sentiment analysis data based on humans interacting with pets, so I used synthetic data (basically, I prompted ChatGPT to give me a bunch of labeled sentiment analysis data)

After that, in order to train the model, I had to pick the best [classifier](https://c3.ai/glossary/data-science/classifier/) to determine which label best fits input given to the model. I went with the [SdcaMaximumEntropy](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.standardtrainerscatalog.sdcamaximumentropy?view=ml-dotnet) classifier, since it's effective for multi-class classification problems like sentiment analysis

### Behavior when ignored
This is part of the wider state system, but the behavior when ignored is one of my favorite parts of the project so I think it should get it's own mini-section

If you pick the action "Ignore" three times in a row, the pet AI will recognize that it's being neglected and will perform an action to try to get attention from the user, like spinning around, jumping, or even whimpering.

It decides the action with an [Epsilon-Greedy Algorithm](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/), which is a common strategy in RL to balance between **exploration** (trying new actions) and **exploitation** (choosing the action with the highest expected reward). The pet AI has a 20% chance to **explore** (choosing a random action) and an 80% chance to **exploit** (choosing the action with the highest Q-value)

The Q-values reflect the expected reward for an action. The better the action is considered in the context of the pet's emotional state and its goal of getting the user's attention.

## Example demonstration

```
Interact with your virtual AI pet! Type "Feed", "Play", "Praise", "Ignore", "Yell", "Take away toy", "Scold" or chat with it. Type "exit" to exit the program.
> Feed
Pet reacts to Feed: Reward 3
Pet state: Neutral | Emotion Score: 3
> Play
Pet reacts to Play: Reward 3.18
Pet state: Happy | Emotion Score: 6
> Praise
Pet reacts to Praise: Reward 5.6
Pet state: Happy | Emotion Score: 11
> You're a good doggo :)
Pet detects sentiment: Positive
Positive score: 88.2% | Neutral score: 0.9% | Negative score: 10.9%
Pet state: Happy | Emotion Score: 14
```
