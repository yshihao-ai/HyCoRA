from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def calculate_bleu_score(generated, ground_truths, lang='zh'):
    try:
        if lang == 'zh':
            ground_truths = [' '.join(list(g)) for g in ground_truths]
            generated = ' '.join(list(generated))
            reference = [list(g.split()) for g in ground_truths]
            candidate = list(generated.split())
        else:
            reference = [word_tokenize(g) for g in ground_truths]
            candidate = word_tokenize(generated)
        bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)
                               , smoothing_function=SmoothingFunction().method3)
        bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)
                               , smoothing_function=SmoothingFunction().method3)
        bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)
                               , smoothing_function=SmoothingFunction().method3)
        bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)
                               , smoothing_function=SmoothingFunction().method3)
        return bleu_1, bleu_2, bleu_3, bleu_4
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0, 0, 0


def evaluate_bleu(response, reference, lang='zh'):
    score_list_1, score_list_2, score_list_3, score_list_4 = [], [], [], []
    for res, ref in zip(response, reference):
        result = calculate_bleu_score(res, ref, lang)
        score_list_1.append(result[0])
        score_list_2.append(result[1])
        score_list_3.append(result[2])
        score_list_4.append(result[3])
    return {
        "bleu-1": sum(score_list_1) / len(score_list_1),
        "bleu-2": sum(score_list_2) / len(score_list_2),
        "bleu-3": sum(score_list_3) / len(score_list_3),
        "bleu-4": sum(score_list_4) / len(score_list_4)
    }


if __name__ == '__main__':
    inference = ["hello there general kenobi", "foo bar foobar a"]  # source
    target = [["hello there general kenobi", "hello there!"], ["foo bar foobar a"]]  # target
    print(evaluate_bleu(inference, target, lang='en'))

    source = r'你好！'  # source
    target = ['你好！', '你好！']  # target
    inference = '你好！'  # inference
    print(evaluate_bleu([inference], [target], lang='zh'))
