'''
Creation date: 7/6/2022

Author: Farzaneh

Modification history: No modification
'''

class Options:
    def __init__(self):
        
        
        #directory settings
        
        self.filepath = 'ExtraSensory.csv'
        self.resultpath='Extrasensory_cleaned2.csv'
        
        
        # Variables
        
        # subject/user id column name
        self.unique_id_variable = ['uuid']  
       
    
        # label type --> category or binary
        self.label_type='binary'  
        self.label_converter='cat_encoding'       
        
        '''
        options:
        'Indicator' 
        'Cat_encoding'
        '''
        
        
        # Category label column name
        self.category_label_variables= []  
        
        
        # Binary label columns names
        self.binary_label_variables = ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking','label:FIX_running', 'label:BICYCLING',
        'label:SLEEPING','label:LAB_WORK', 'label:IN_CLASS', 'label:IN_A_MEETING','label:LOC_main_workplace', 'label:OR_indoors',
        'label:OR_outside','label:IN_A_CAR', 'label:ON_A_BUS', 'label:DRIVE_-_I_M_THE_DRIVER','label:DRIVE_-_I_M_A_PASSENGER',
        'label:LOC_home','label:FIX_restaurant', 'label:PHONE_IN_POCKET', 'label:OR_exercise','label:COOKING', 'label:SHOPPING',
        'label:STROLLING','label:DRINKING__ALCOHOL_', 'label:BATHING_-_SHOWER', 'label:CLEANING','label:DOING_LAUNDRY', 
        'label:WASHING_DISHES', 'label:WATCHING_TV','label:SURFING_THE_INTERNET', 'label:AT_A_PARTY', 'label:AT_A_BAR', 'label:LOC_beach',
        'label:SINGING', 'label:TALKING','label:COMPUTER_WORK', 'label:EATING', 'label:TOILET', 'label:GROOMING','label:DRESSING',
        'label:AT_THE_GYM', 'label:STAIRS_-_GOING_UP', 'label:STAIRS_-_GOING_DOWN', 'label:ELEVATOR', 'label:OR_standing','label:AT_SCHOOL',
        'label:PHONE_IN_HAND', 'label:PHONE_IN_BAG', 'label:PHONE_ON_TABLE', 'label:WITH_CO-WORKERS', 
        'label:WITH_FRIENDS']
        
        
        # Timestamp variables (binary and non_binary)
        self.timestamp_variables = ['timestamp','discrete:time_of_day:between0and6','discrete:time_of_day:between3and9',
       'discrete:time_of_day:between6and12', 'discrete:time_of_day:between9and15', 'discrete:time_of_day:between12and18',
       'discrete:time_of_day:between15and21', 'discrete:time_of_day:between18and24', 'discrete:time_of_day:between21and3']
        
        
        # Extra features
        self.extra_features=['label_source']
        

        #Impute missing values
        
        # Imputer method for features
        self.miss_imputer_feat='Filling zero'
        
        # Imputer method for labels
        self.miss_imputer_lab='Filling zero'
           
            
       # Missing value imputation options:     
        '''
        Numerical features -->
        - 'Filling zero'
        - 'Mean imputation'
        - 'Median imputation'
        - 'Linear interpolation'
        '''
        
        
        #Outlier detection quantile (upper bound and lower bound range)
        self.quantile=0.025
        
        
        #Scaling methods (Numerical features)
        self.scaling_method= 'Normalization'
        
        
        # Scaling method options:
        '''
        - 'Normalization'               #Scales the data using the formula (x - min)/(max - min)
        - 'Standardization'             #Scales the data using the formula (x-mean)/standard deviation
        '''
        
        
        
        
        