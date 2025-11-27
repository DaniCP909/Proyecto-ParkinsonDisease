from domain.RepresentationType import RepresentationType
from domain.Task import Task

class Patient:

    def __init__(self, patient_id: int, patient_disease_status:int, patient_disease_years:int):
        self.id = patient_id
        self.pd_status = patient_disease_status
        self.pd_years = patient_disease_years


        self.tasks_lists_dict = {
            RepresentationType.SIMPLE_STROKE: [],
            RepresentationType.ENHANCED_STROKE: [],
            RepresentationType.MULTICHANNEL: [],
            RepresentationType.ONLINE_SIGNAL: [],
        }
        

    def getId(self):
        return self.id
    
    def setId(self, id):
        self.id = id

    def getPdStatus(self):
        return self.pd_status
    
    def setPdStatus(self, pd_status):
        self.pd_status = pd_status

    def getYears(self):
        return self.pd_years
    
    def setYears(self, years):
        self.pd_years = years

    def addTask(self, new_task: Task):
        rep_type = new_task.getRepType()
        if rep_type not in self.tasks_lists_dict:
            raise ValueError("Unknown representation type")
        self.tasks_lists_dict[rep_type].append(new_task)

    def getTasksListsDict(self):
        return self.tasks_lists_dict
        
    def getTasksByType(self, rep_type: RepresentationType):
        return self.tasks_lists_dict[rep_type]
    
    def getTaskByTypeAndNum(self, rep_type: RepresentationType, task_num: int):
        tasks = self.tasks_lists_dict[rep_type]
        index = task_num - 1

        if index < 0 or index >= len(tasks):
            raise ValueError(f"No task {task_num} for subject {self.id}")
        
        return tasks[index]