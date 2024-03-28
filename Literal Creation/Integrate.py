import Agent_MS
import Retrival_model
import Evaluate
import json
from tqdm import tqdm

if __name__ == '__main__':
    Retrival_model.main("./LiteralTrain.jsonl")

    """
    while True:
        all_sentences = Evaluate.allSentences("./wuyanlvshi.jsonl")
        user_input = input("Question:")
        agent_type = input("Select one from: Literature, Logic, Translate")
        final_question = Evaluate.final_prompt(user_input, "model.pth", "optimizer.pth", all_sentences)
        answer = Evaluate.chatgpt_answer(final_question)
        Agent_MS.message_store(final_question, answer, "./train.jsonl", agent_type)
        print(final_question)
        print(answer)
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break  # This exits the loop #------------------------------------
    """
    problem_list = []
    with open("./Literalproblem.jsonl", 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            problem_list.append(data["messages"][0]["content"])
    count = 0
    total_count = 0
    for problem in tqdm(problem_list):
        all_sentences = Evaluate.allSentences("./LiteralPool.jsonl")
        final_question = Evaluate.final_prompt(problem, "model.pth", "optimizer.pth", all_sentences)
        answer = Evaluate.chatgpt_answer(final_question)
        totot_count = total_count + 1
        bValue = Agent_MS.message_store(final_question, answer, "./LiteralPool.jsonl", "Literature")
        if bValue:
            count = count + 1
        if count % 20 == 0:
            print("total_Count:" + str(total_count))
            print("Count:" + str(count))
            print("For Limerick: ----------------------------")
            Evaluate.main("LiteralTrain.jsonl", "limerickTest.jsonl")
            print("For wuyanlvshi:--------------------------")
            Evaluate.main("LiteralTrain.jsonl", "lvshiTest.jsonl")
            print("For sonnet")
            Evaluate.main("LiteralTrain.jsonl", "sonnetTest.jsonl")

