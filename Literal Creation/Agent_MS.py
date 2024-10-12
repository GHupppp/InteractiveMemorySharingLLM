import openai
import os
from collections import defaultdict
from openai import OpenAI
import json
import Retrival_model
import torch #

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


dic_grading = {"Literature": """
General Evaluation Criteria (Total: 100 Points)

Criteria: Literary Quality
Score Range: 0-5
Description: Assesses creativity, use of language, and emotional impact. High-quality examples should demonstrate mastery of language and evoke a strong reader response.

Criteria: Authenticity
Score Range: 0-10
Description: Evaluates adherence to the form's traditional standards, including structure, rhythm, and themes. High scores indicate that the poem respects genre conventions creatively.

Criteria: Clarity and Cohesion
Score Range: 0-10
Description: Considers the poem's clarity of expression and the cohesion of its parts. A high score indicates that the poem communicates effectively and its elements are well integrated.

Criteria: Innovativeness
Score Range: 0-5
Description: Rewards originality in theme, structure, or language use. High scores reflect a notable degree of creativity and the introduction of novel ideas or techniques.

Criteria: Educational Value
Score Range: 0-10
Description: Assesses the example's potential to teach about poetic forms, literary devices, and thematic exploration. High-scoring examples are rich in analyzable and teachable elements.

Criteria: Metric Precision
Score Range: 0-10
Description: Evaluates the adherence to the five-syllable structure per line, including rhythm and flow, emphasizing the importance of metric accuracy.

Criteria: Imagery and Symbolism
Score Range: 0-10
Description: Assesses the effectiveness of imagery and symbolism in conveying the poem's themes, highlighting the depth and sophistication of language use.

Criteria: Humor and Wit
Score Range: 0-10
Description: Rates the poem's humor, wit, and wordplay. High scores reflect effective use of language to entertain and amuse.

Criteria: Rhyme Scheme Adherence
Score Range: 0-10
Description: Assesses the AABBA rhyme scheme's quality and creativity, including how well the rhymes enhance the humor and effectiveness of the poem.

Criteria: Structural Integrity
Score Range: 0-10
Description: Evaluates adherence to sonnet structure, including rhyme scheme and division into octaves/sestets or quatrains/couplet, stressing formal precision.

Criteria: Thematic Development
Score Range: 0-10
Description: Looks at theme or argument development, especially through the volta, reflecting the poem's ability to engage with complex ideas persuasively.
""", "Logic": """
1. Clarity and Understandability (20 points)
Question Clarity (10 points): The question should be clearly stated, without ambiguity, and understandable without requiring additional context.
Answer Clarity (10 points): The answer should be directly related to the question, clear, and easily understandable.
2. Creativity and Originality (30 points)
Question Creativity (15 points): The question should demonstrate creativity, originality, and should not be a common or easily found problem.
Answer Creativity (15 points): The answer should be innovative and not just a straightforward or commonly known response. It should also add a layer of depth or a surprising twist to the question.
3. Logical Consistency and Correctness (20 points)
Logical Consistency (10 points): The question and answer together should form a logically consistent pair where the answer correctly follows from the question.
Correctness (10 points): The answer must be factually correct and provide a true solution or conclusion to the puzzle, riddle, or pun presented in the question.
4. Relevance and Engagement (20 points)
Relevance (10 points): The question and answer should be relevant to the domain of Logic Problems, demonstrating an understanding of puzzles, riddles, or puns.
Engagement (10 points): The pair should be engaging and interesting, capable of capturing attention and sparking curiosity or amusement.
5. Difficulty Level (10 points)
The difficulty level of the question should be appropriate for the intended audience. It should neither be too easy to solve without any thought nor too difficult to be practically unsolvable. This criterion requires a balanced approach to ensure the content is intellectually stimulating but accessible.
""", "Plan": """
1. Specificity and Detail (20 points)
Question Specificity (10 points): The question should be specific, providing enough detail to guide the generation of a relevant and tailored plan.
Plan Detail (10 points): The plan should include specific activities, steps, or recommendations that are clearly defined and actionable.
2. Feasibility and Practicality (20 points)
Plan Feasibility (20 points): The plan should be realistic and practical, considering available resources (time, money, equipment) and constraints. It should propose actions that can be realistically implemented by the user.
3. Comprehensiveness and Scope (20 points)
Coverage of Key Components (20 points): The plan should comprehensively address all relevant aspects of the goal. For a study plan, this might include study sessions, breaks, and topics covered; for a fitness plan, workouts, rest days, and nutrition; and for a travel plan, transportation, accommodations, and activities.
4. Personalization and Relevance (20 points)
Alignment with User Needs and Preferences (20 points): The plan should reflect an understanding of the user's specific needs, preferences, goals, and limitations. It should feel customized and directly applicable to the user, rather than being a generic template.
5. Clarity and Understandability (20 points)
Plan Clarity (10 points): The plan should be articulated in a clear, organized, and easy-to-follow manner. It should avoid jargon or overly complex language, making it accessible to the user.
Rationale Clarity (10 points): The plan should include clear reasoning or justification for the recommendations made, helping the user understand why specific actions or steps are suggested.
""", "Total": """1. Accuracy (25 Points)
25 points: The output is entirely accurate, with no factual errors or inaccuracies.
15-24 points: The output is mostly accurate, with minor errors that do not significantly impact the overall understanding.
5-14 points: The output contains several inaccuracies that could lead to misunderstandings.
0-4 points: The output is largely inaccurate, misleading, or irrelevant.
2. Relevance (20 Points)
20 points: The output is highly relevant to the input question, directly addressing the query without diverging from the topic.
10-19 points: The output is relevant but includes some unnecessary or slightly off-topic information.
1-9 points: The output partially addresses the question but is significantly off-topic or tangential.
0 points: The output is completely irrelevant to the input question.
3. Completeness (20 Points)
20 points: The output provides a complete answer to the question, covering all essential aspects implied or directly asked.
10-19 points: The output covers most of the necessary information but lacks one or two minor details or aspects.
1-9 points: The output provides a partial answer, missing significant portions of the information needed to fully answer the question.
0 points: The output fails to provide any meaningful answer to the question.
4. Clarity and Coherence (20 Points)
20 points: The output is exceptionally clear and well-structured, making it easy to follow and understand.
10-19 points: The output is clear but may have minor issues with structure or coherence that slightly hinder understanding.
1-9 points: The output has significant clarity or coherence issues, making it difficult to understand without effort.
0 points: The output is incoherent or so poorly structured that it is unintelligible.
5. Creativity and Insight (15 Points)
15 points: The output demonstrates high levels of creativity or provides insights that add substantial value beyond the explicit question.
8-14 points: The output shows some creativity or insights but to a lesser extent, offering added value to the answer.
1-7 points: The output is standard, with minimal to no creativity or insightful additions.
0 points: The output is entirely generic, with no attempt at creativity or providing additional insights."""}


def valid_fine_tuning(file_name):
    data_path = file_name

    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


def scoring_by_chatgpt(question, answer, grading_category):
    os.environ["OPENAI_API_KEY"] = "..."
    openai.api_key = os.environ["OPENAI_API_KEY"]
    grading_rubric = dic_grading[grading_category]
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Agent fouce on grading the example for In-context learning, "
                                          "if the given example is good for In-context learning, give the example a "
                                          "high score, otherwise, a low score"},
            {"role": "user", "content": f"Here is the rubrics for grading an example of In-context learning-{grading_rubric}. "
                                        f"According to the rubric, for the Question-{question} and Answer-{answer}, "
                                        f"give me a score of it if I want to use it as a prompt in In-context "
                                        f"learning later. Give me a number between 0-100.For your answer, just give me "
                                        f"the socre(a number), no other thing. "
                                        f"Remember, the output you give to me can only be a number, no other words! "
                                        f"If it is not a number, my whole project will be destoryed"}
        ]
    )
    return completion.choices[0].message.content


def train_model_duringStore(question, answer, file_name):
    model = Retrival_model.SentenceSimilarityModel()
    optimizer = Retrival_model.AdamW(model.parameters(), lr=5e-5)
    # Load the saved states
    Retrival_model.load_model_and_optimizer(model, optimizer, "model.pth", "optimizer.pth")
    model = model.to(device)
    list1 = Retrival_model.allSentences(file_name)
    question_combine = f"{question}<->{answer}"
    list2 = Retrival_model.retrieval_first_BM25(question_combine, list1)
    sentence_pairs, labels = Retrival_model.grade_and_select_forMemory(question_combine, list2)
    Retrival_model.training_model(model, sentence_pairs, labels, optimizer, epochs=2)
    Retrival_model.save_model_and_optimizer(model, optimizer, "model.pth", "optimizer.pth")
    print("Single-memory train is done")


# The item in the file is decrease according to the scores, there will be a score made by chatgpt in advance
def message_store(question, answer, file_path, grading_category):
    score = scoring_by_chatgpt(question, answer, grading_category)
    try:
        score = int(score)  # Attempt to convert
    except ValueError:
        print("Not a good example for In-Context learning")
        return
    if score < 50:
        print("Not a good example for In-Context learning. Score is too low")
        print(score)
        return
    message_structure = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }

    train_model_duringStore(question, answer, file_path)

    try:
        with open(file_path, 'a', encoding='utf-8') as write_file:
            write_file.write(json.dumps(message_structure, ensure_ascii=False) + '\n')
            #print("Successfully store one memory")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


'''
# The item in the file is decrease according to the scores, there will be a score made by chatgpt in advance
def message_store(question, answer, file_path):
    score = scoring_by_chatgpt(question, answer)
    try:
        score = int(score)  # Attempt to convert
    except ValueError:
        print("Not a good example for In-Context learning")
        return
    if score < 7:
        print("Not a good example for In-Context learning. Score is too low")
        return
    message_structure = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "scores": score
    }

    train_model_duringStore(question, answer)

    try:
        inserted = False
        temp_path = file_path + ".tmp"
        with open(file_path, 'r', encoding='utf-8') as read_file, open(temp_path, 'w', encoding='utf-8') as write_file:
            for line in read_file:
                existing_dict = json.loads(line)
                # Assuming 'scores' is always present and properly formatted
                if not inserted and existing_dict.get("scores", -1) < message_structure.get("scores", -1):
                    write_file.write(json.dumps(message_structure, ensure_ascii=False) + '\n')
                    inserted = True
                write_file.write(line)
            # If new_dict has the highest 'scores' or the file is empty
            if not inserted:
                write_file.write(json.dumps(message_structure, ensure_ascii=False) + '\n')
        # Replace the old file with the new file
        os.replace(temp_path, file_path)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
'''
