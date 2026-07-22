from rouge_score import rouge_scorer
import sys
from typing import List

def evaluate_rouge(generated_text: str, reference_text: str) -> float:
    """
    Evaluate the ROUGE-L F1 score between generated and reference text.
    
    Args:
        generated_text: The output from the LLM.
        reference_text: The ground truth reference answer.
        
    Returns:
        float: ROUGE-L F1 score.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure

def evaluate_bertscore(generated_texts: List[str], reference_texts: List[str]) -> List[float]:
    """
    Evaluate the BERTScore F1 between lists of generated and reference texts.
    Requires bert_score to be installed.
    
    Args:
        generated_texts: List of outputs from the LLM.
        reference_texts: List of ground truth reference answers.
        
    Returns:
        List[float]: BERTScore F1 scores for each pair.
    """
    try:
        from bert_score import score
    except ImportError:
        print("Warning: bert_score is not installed. Returning 0.0 for BERTScore.", file=sys.stderr)
        return [0.0] * len(generated_texts)

    # Calculate BERTScore
    P, R, F1 = score(generated_texts, reference_texts, lang="en", verbose=False)
    return F1.tolist()
