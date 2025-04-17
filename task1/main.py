from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Task(BaseModel):
    title: str
    description: Optional[str] = None
    status: str

tasks = {}
task_id_counter = 1

@app.get("/tasks", response_model=List[Task])
def get_tasks():
    return list(tasks.values())

@app.post("/tasks", response_model=Task)
def create_task(task: Task):
    global task_id_counter
    task_id = task_id_counter
    tasks[task_id] = task
    task_id_counter += 1
    return task

@app.put("/tasks/{task_id}", response_model=Task)
def update_task(task_id: int, task: Task):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    tasks[task_id] = task
    return task

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    del tasks[task_id]
    return {"detail": "Task deleted"}