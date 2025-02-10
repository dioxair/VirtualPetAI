using VirtualPetAI.ML;

internal class Program
{
    static void Main(string[] args)
    {
        VirtualPet pet = new();
        Console.WriteLine("Interact with your virtual AI pet! Type \"Feed\", \"Play\", \"Praise\", \"Ignore\", \"Yell\", \"Take away toy\", \"Scold\" or chat with it. Type \"exit\" to exit the program.");

        while (true)
        {
            Console.Write("> ");
            string input = Console.ReadLine();
            if (input == "exit") break;
            if (input == "Feed" || input == "Play" || input == "Scold" || input == "Praise" || input == "Yell" ||
                input == "Take away toy")
                pet.PerformAction(input);
            else
                pet.ChatWithPet(input);
        }
    }
}
