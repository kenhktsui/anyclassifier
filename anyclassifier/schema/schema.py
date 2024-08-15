from typing import List
from pydantic import BaseModel, RootModel


class Item(BaseModel):
    item: str


ItemList = RootModel[List[Item]]


class SourceType(BaseModel):
    source_type: str


SourceTypeList = RootModel[List[SourceType]]


class Record(BaseModel):
    text: str


SyntheticData = RootModel[List[Record]]


class Label(BaseModel):
    """
    id is used for `label` in the dataset
    desc is used in the prompt
    """
    id: int
    desc: str


class Example(BaseModel):
    text: str
    label: int