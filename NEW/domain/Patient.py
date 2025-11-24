

class Patient:

    def __init__(self, patient_id: int, patient_disease_status:int, patient_disease_years:int):
        self.id = patient_id
        self.pd_status = patient_disease_status
        self.pd_years = patient_disease_years

        self.simple_stroke_tasks = []
        self.enhanced_stroke_tasks = []
        self.online_signal_tasks = []
        self.multichannel_tasks = []