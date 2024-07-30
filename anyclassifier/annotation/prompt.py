from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Label:
    name: str
    desc: str


@dataclass
class Example:
    text: str
    label: str


@dataclass
class AnnotationPrompt:
    task_description: str
    label_definition: List[Label]
    few_shot_examples: Optional[List[Example]] = None

    def get_prompt(self, text: str):
        label_defn_str = "\n".join([f"{ld.name}: {ld.desc}" for ld in self.label_definition])
        example_str = "" if self.few_shot_examples is None else "\nHere are some examples:\n" + "\n".join(
            [f"Example {i+1}.\nText: {fse.text}\nLabel: {fse.label}" for i, fse in enumerate(self.few_shot_examples)]
        )
        return f"""{self.task_description}
Here are the label definitions:
{label_defn_str}

Here is the text to be analyzed:
<text>
{text}
</text>
{example_str}
Provide your analysis in the following format:
<analysis>
Label: [Your answer]
</analysis>"""


if __name__ == "__main__":
    prompt = AnnotationPrompt(
        task_description="Classify a text's sentiment.",
        label_definition=[
            Label(name="1", desc='positive sentiment'),
            Label(name="0", desc='negative sentiment')
        ],
        few_shot_examples=[
            Example(text="This is good movie", label="positive"),
            Example(text="It is a waste of time.", label="negative")
        ]
    )
    print(prompt.get_prompt("It is one of best I ever watched."))
    prompt = AnnotationPrompt(
        task_description="Classify a text's sentiment.",
        label_definition=[
            Label(name="1", desc='positive sentiment'),
            Label(name="0", desc='negative sentiment')
        ]
    )
    print(prompt.get_prompt("It is one of best I ever watched."))
