from typing import List, Optional, Union, Tuple, Dict, Any
from core.utils.common import (
    prepare_multiturn_grounding_questions,
)

### Preprocess/Postprocess Helpers ###
def prepare_target_objects_questions(
    task_descriptions: List[str]
) -> List[str]:
    assert isinstance(task_descriptions, list) or isinstance(task_descriptions, str), \
        "task_descriptions must be a list of strings or a single string."
    if isinstance(task_descriptions, str):
        task_descriptions = [task_descriptions]
    target_objects_questions = []
    for desc in task_descriptions:
        question = (
            f"Task description: {desc}\n"
            "Instruction: Find the operated objects and the target objects in the task description. "
            "Give the full description of the objects, including color, shape, and other attributes. "
            "If one of the objects is missing, you can just output the existing objects.\n"
            "Output the results as a list of strings. No other text.\n"
            "Example:\n"
            "  Input: put the green apple in the pink plate"
            "  Output: ['green apple', 'pink plate']\n"
            # "  Input: push the block with number 2 to the front"
            # "  Output: ['block with number 2']\n"
            # "  Input: push i"
            # "  Output: ['block with letter i']\n"
        )
        target_objects_questions.append(question)
    return target_objects_questions