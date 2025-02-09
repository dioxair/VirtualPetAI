using VirtualPetAI.ML;

internal class Program
{
    static void Main(string[] args)
    {
        VirtualPet pet = new();
        Console.WriteLine("Interact with your virtual AI pet! Type \"Feed\", \"Play\", \"Scold\" or chat with it. Type \"exit\" to exit the program.");

        while (true)
        {
            string input = Console.ReadLine();
            if (input == "exit") break;
            if (input == "Feed" || input == "Play" || input == "Scold")
                pet.PerformAction(input);
            else
                pet.ChatWithPet(input);
        }
    }
}
