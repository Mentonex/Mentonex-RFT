# -*- coding: utf-8 -*-
"""Schema for different types of data."""
from typing import Any, Optional, Type

from sqlalchemy import JSON, Column, DateTime, Float, Integer, LargeBinary, String, Text
from sqlalchemy.ext.declarative import declarative_base

from trinity.common.experience import Experience
from trinity.common.models.utils import tokenize_and_mask_messages_hf

Base: Type = declarative_base()

# TODO: create db engine and all tables in a factory class


class RftDatasetModel(Base):
    """SQLAlchemy model for RftDataset."""

    __tablename__ = "rft_dataset"

    # lineage
    id = Column(Integer, primary_key=True, autoincrement=True)
    consumed_cnt = Column(Integer, default=0)
    last_modified_date = Column(DateTime, nullable=True)
    from_id = Column(Integer, nullable=True)
    from_model = Column(Text, nullable=True)
    from_recipe = Column(Text, nullable=True)
    # content
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    solution = Column(Text, nullable=True)
    reward = Column(Float, nullable=True)
    chosen = Column(Text, nullable=True)
    rejected = Column(Text, nullable=True)
    label = Column(Text, nullable=True)
    # extra info
    quality_score = Column(Float, default=0.0)
    quality_score_detail = Column(JSON, nullable=True)
    difficulty_score = Column(Float, default=0.0)
    difficulty_score_detail = Column(JSON, nullable=True)
    diversity_score = Column(Float, default=0.0)
    diversity_score_detail = Column(JSON, nullable=True)
    priority = Column(Float, default=0.0)
    # downstream
    reward_fn = Column(Text, nullable=True)
    workflow = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {key: val for key, val in self.__dict__.items() if not key.startswith("_")}


class TaskModel(Base):
    """SQLAlchemy model for Task."""

    # TODO: Add more fields

    __tablename__ = "task_buffer"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_desc = Column(String, nullable=True)
    workflow_type = Column(String, nullable=True)
    reward_type = Column(String, nullable=True)


class ExperienceModel(Base):
    """SQLAlchemy model for Experience."""

    __tablename__ = "experience_buffer"

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    prompt = Column(String, nullable=True)
    response = Column(String, nullable=True)
    reward = Column(Float, nullable=True)
    consumed = Column(Integer, default=0)
    priority = Column(Float, default=0.0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.serialized_exp)

    @staticmethod
    def from_experience(experience: Experience):
        """Save the experience to database."""
        return ExperienceModel(
            serialized_exp=experience.serialize(),
            reward=experience.reward,
            prompt=experience.prompt_text,
            response=experience.response_text,
        )


class SFTDataModel(Base):
    """SQLAlchemy model for SFT data."""

    __tablename__ = "sft_data_buffer"

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    messages = Column(String, nullable=True)
    consumed = Column(Integer, default=0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.serialized_exp)

    @classmethod
    def from_messages(
        cls,
        messages: list[dict],
        tokenizer: Any,
        chat_template: Optional[str] = None,
    ) -> "SFTDataModel":
        """Convert a list of messages into a single instance of SFT data."""
        token_ids, action_mask = tokenize_and_mask_messages_hf(
            tokenizer=tokenizer,
            messages=messages,
            chat_template=chat_template,
        )
        exp = Experience(
            tokens=token_ids,
            prompt_length=0,
            action_mask=action_mask,
            info={"response_num": sum([1 if m["role"] == "assistant" else 0 for m in messages])},
        )
        return cls(
            serialized_exp=exp.serialize(),
            messages=messages,
        )


class DPODataModel(Base):
    """SQLAlchemy model for DPO data."""

    __tablename__ = "dpo_data_buffer"

    id = Column(Integer, primary_key=True, autoincrement=True)
    serialized_exp = Column(LargeBinary, nullable=True)
    chosen = Column(LargeBinary, nullable=True)
    rejected = Column(LargeBinary, nullable=True)
    consumed = Column(Integer, default=0)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        exp = Experience.deserialize(self.serialized_exp)
        exp.chosen = Experience.deserialize(self.chosen)
        exp.rejected = Experience.deserialize(self.rejected)
        return exp
