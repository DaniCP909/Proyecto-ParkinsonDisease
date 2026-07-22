from domain.RepresentationType import RepresentationType
from domain.Task import Task

class EmothawPatient:

    def __init__(self, patient_id: int, depression_score: int, anxiety_score: int, stress_score: int):

        self.id = patient_id
        
        self.depression_score = depression_score
        self.anxiety_score = anxiety_score
        self.stress_score = stress_score

        self.tasks_dicts_dict = {
            RepresentationType.SIMPLE_STROKE: {},
            RepresentationType.ENHANCED_STROKE: {},
            RepresentationType.MULTICHANNEL: {},
            RepresentationType.ONLINE_SIGNAL: {},
        }

    def getId(self):
        return self.id
    
    def getDepressionScore(self):
        return self.depression_score
    
    def getAnxietyScore(self):
        return self.anxiety_score
    
    def getStressScore(self):
        return self.stress_score
    
    def addTask(self, new_task: Task):
        rep_type = new_task.getRepType()
        if rep_type not in self.tasks_dicts_dict:
            raise ValueError("Unknown representation type")
        self.tasks_dicts_dict[rep_type][new_task.task_number] = new_task
    def getTasksListsDict(self):
        return self.tasks_dicts_dict
        
    def getTasksByType(self, rep_type: RepresentationType):
        return self.tasks_dicts_dict[rep_type]
    
    def getTaskByTypeAndNum(self, rep_type: RepresentationType, task_num: int):
        task = self.tasks_dicts_dict[rep_type][task_num]

        return task
    
    def getTaskNumbers(self):
        return list(self.tasks_dicts_dict[RepresentationType.SIMPLE_STROKE].keys())