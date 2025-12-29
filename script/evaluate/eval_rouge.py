from rouge import Rouge

rouge = Rouge()


def calculate_rouge_score(generated, ground_truths, lang='zh'):
    try:
        rouge_1_r, rouge_1_f, rouge_1_p = 0, 0, 0
        rouge_2_r, rouge_2_f, rouge_2_p = 0, 0, 0
        rouge_l_r, rouge_l_f, rouge_l_p = 0, 0, 0

        for ground_truth in ground_truths:
            if lang == 'zh':
                score = rouge.get_scores(' '.join(list(generated)), ' '.join(list(ground_truth)))
            else:
                score = rouge.get_scores(generated, ground_truth)
            
            rouge_1_f = max(score[0]['rouge-1']['f'], rouge_1_f)
            rouge_1_r = max(score[0]['rouge-1']['r'], rouge_1_r)
            rouge_1_p = max(score[0]['rouge-1']['p'], rouge_1_p)

            rouge_2_f = max(score[0]['rouge-2']['f'], rouge_2_f)
            rouge_2_r = max(score[0]['rouge-2']['r'], rouge_2_r)
            rouge_2_p = max(score[0]['rouge-2']['p'], rouge_2_p)
            
            rouge_l_f = max(score[0]['rouge-l']['f'], rouge_l_f)
            rouge_l_r = max(score[0]['rouge-l']['r'], rouge_l_r)
            rouge_l_p = max(score[0]['rouge-l']['p'], rouge_l_p)

        return {
                "rouge-1": (rouge_1_r, rouge_1_p, rouge_1_f),
                "rouge-2": (rouge_2_r, rouge_2_p, rouge_2_f),
                "rouge-l": (rouge_l_r, rouge_l_p, rouge_l_f),
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return {
                "rouge-1": (0, 0, 0),
                "rouge-2": (0, 0, 0),
                "rouge-l": (0, 0, 0),
        }


def evaluate_rouge(response, reference, lang='zh'):
    score_list_r_1, score_list_p_1, score_list_f_1 = [], [], []
    score_list_r_2, score_list_p_2, score_list_f_2 = [], [], []
    score_list_r_l, score_list_p_l, score_list_f_l = [], [], []
    for res, ref in zip(response, reference):
        score = calculate_rouge_score(res, ref, lang=lang)
        score_list_r_1.append(score['rouge-1'][0])
        score_list_p_1.append(score['rouge-1'][1])
        score_list_f_1.append(score['rouge-1'][2])

        score_list_r_2.append(score['rouge-2'][0])
        score_list_p_2.append(score['rouge-2'][1])
        score_list_f_2.append(score['rouge-2'][2])

        score_list_r_l.append(score['rouge-l'][0])
        score_list_p_l.append(score['rouge-l'][1])
        score_list_f_l.append(score['rouge-l'][2])

    return {
            "rouge-1-r": sum(score_list_r_1) / len(score_list_r_1),
            "rouge-1-p": sum(score_list_p_1) / len(score_list_p_1),
            "rouge-1-f": sum(score_list_f_1) / len(score_list_f_1),

            "rouge-2-r": sum(score_list_r_2) / len(score_list_r_2),
            "rouge-2-p": sum(score_list_p_2) / len(score_list_p_2),
            "rouge-2-f": sum(score_list_f_2) / len(score_list_f_2),

            "rouge-l-r": sum(score_list_r_l) / len(score_list_r_l),
            "rouge-l-p": sum(score_list_p_l) / len(score_list_p_l),
            "rouge-l-f": sum(score_list_f_l) / len(score_list_f_l),
    }
