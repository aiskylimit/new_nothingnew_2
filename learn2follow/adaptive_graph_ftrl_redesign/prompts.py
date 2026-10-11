"""Question-only student and privileged reference teacher templates."""

def student_user_message(problem: str) -> str:
    return f"Problem: {problem.strip()}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

def teacher_user_message(problem: str, reference_solution: str) -> str:
    return (
        f"Problem: {problem.strip()}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{reference_solution.strip()}\n"
        "=== Reference Solution End ===\n\n"
        "After reading the reference solution above, use your own words and independent reasoning "
        "to derive the same final answer. Think step by step, explore alternatives, and reconsider "
        "an approach if it does not work.\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
