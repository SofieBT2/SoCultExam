# FUNCTIONS FOR OUR ABM

# 1 tick = 1 week
# i alt: 52*5: 260 ticks

# defining agents as abstract class
class Agent:
    # init-method, the constructor method for agents
    # maybe we dont need the position and index parameters? these will be given in the populate_company function
    def __init__(self, age, gender, symptom_score, concussion_score, energy_level, id):
        self.age = age
        self.gender = gender
        self.symptom_score = symptom_score
        self.concussion_score = concussion_score
        self.energy_level = energy_level
        self.id = id

# function for creating empty dictionary (Labour market in country with x concussion treatment)
def create_labourmarket(attachment:list, n:list):
    '''
    Creates a dictionary with the given type of labour market attachment as keys and empty lists as values.

    Parameters
    ----------
    labour market attachment : list, a list of strings with type of labour market attachment
    n : list, a list of integers with the number of agents in each type of labour market attachment
    '''
    labourmarket = {}
    for i in range(len(attachment)):
        key = attachment[i] 
        value = [None for i in range(0, n[i])]
        labourmarket[key] = value

    return labourmarket

def get_truncated_normal(mean=0, sd=1, low=20, upp=68):
    x = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return x.rvs(1)[0]

# function for creating the first agents and putting them in the labourmarket: 
# *NOTE: think about what our baseline should be: are we interested in modelling how agents in different groups are affected by concussion or only interested in fulltime agents?
# change attributes to our attributes: 

def populate_labourmarket(labourmarket: dict, weights: dict):
    '''
    Creates the first agents and puts them in the labourmarket dictionary.

    Parameters
    ----------
    labourmarket : dictionary, a dictionary with the attachment as keys and empty lists as values
    weights: dictionary, a dictionary containing information on how to generate the attributes of the agents in each attachment
    '''
    id = 1
    for i in labourmarket.keys():
        for j in range(0, len(labourmarket[i])):
            # id
            id += 1
            labourmarket[i][j] = Agent(position = i, index = j, gender = random.choices(['male', 'female'], weights = weights[i]['weights'], k = 1), age = get_truncated_normal(mean = weights[i]['age'][0], sd = weights[i]['age'][1]), seniority = random.gauss(weights[i]['seniority'][0], weights[i]['seniority'][1]), fire = random.choices([True, False], weights = weights[i]['fire']), seniority_position = random.gauss(weights[i]['seniority_position'][0], weights[i]['seniority_position'][1]), id = id)
    return id

# this function should instead count the numbers of people in each attachment group after x ticks. Should be modified later on
def count_level_agents(labourmarket):
    '''
    Counts the number of level of concussion-agents in each attachment

    Parameters
    ---------- 
    labourmarket : dictionary, a dictionary with the attachment as keys and empty lists as values
    '''
    mild_count = np.repeat(0, len(labourmarket.keys()))
    moderate_count = np.repeat(0, len(labourmarket.keys()))
    severe_count = np.repeat(0, len(labourmarket.keys()))

    for index, i in enumerate(labourmarket.keys()):
        for j in range(0, len(labourmarket[i])):
            if labourmarket[i][j] is not None:
                if labourmarket[i][j].concussion_level == ['mild']:
                    mild_count[index] += 1
                if labourmarket[i][j].concussion_level == ['moderate']:
                    moderate_count[index] += 1
                else:
                    severe_count[index] += 1
    return mild_count, moderate_count, severe_count

def agent_movement (...): # function for defining when an agent is moved from one type of attachment to another
    return()

def agent_out (): # nødvendig? if an agent is completely detached from the labour market (making a seperate function, as the agent should be pulled out of the tick loop then)
    return()

def symptom_score (): # for calculating the symptom score, columns = symptoms, level of symptom, decay_rate
    return()

def concussion_level (): # for calculating the level of concussion (based on symptom score)
    return(concussion_level)

def energy_level (): # for calculating the level of energy (based on a initial energy distribution - daily task - symptoms)
    return(energy_level)

def update_agent(labourmarket: dict, i, j, months_pl): # updating agents after each tick
    '''

    Parameters
    ----------
    labourmarket : dictionary
    i : the attachment of the 
    j : the index of the agent at the job-title
    '''
    return ()

# Definition for running the ABM
def run_abm(months: int, save_path: str, company_titles: list, titles_n: list, weights: dict, bias_scaler: float = 1.0, plot_each_tick = False, months_pl: float = 9, threshold: float = 0.3, diversity_bias_scaler: float = 1.0, intervention = None, company = None):
    '''
    Runs the ABM simulation

    Parameters
    ----------
    months : int, the number of months to simulate
    save_path : str, the path to the csv file
    company_titles : list, a list of strings with the job titles
    titles_n : list, a list of integers with the number of agents in each job title
    plot_each_tick : bool, if True, the gender distribution is plotted after each tick
    weights : dictionary, a dictionary containing the information used to generate agents
    bias_scaler : float, higher number increases the influence of the bias of the gender distribution at the level at which a position is empty
    month_pl : int, the number of months women are on parental leave after giving birth
    threshold : list, the threshold for the share of women at a given level (if below threshold a positive bias is added towards women, i.e., increasing their probability of promotion)
    intervention : the type of intervention to apply
    '''
    # creating empty dataframe for the results
    data = create_dataframe(['tick', 'gender'], company_titles)
    adata = pd.DataFrame()
    
    # create company
    if company == None:
        company = create_company(company_titles, titles_n)
        # populate company using weights, and saving the last ID given to an agent
        id = populate_company(company, weights)
    else: 
        id = 0

    # plot initial
    if plot_each_tick:
        plot_gender(company, tick=-1)

    # iterating though the months
    for month in range(months):
        bias = list(get_bias(company, scale = bias_scaler))
        mean_senior = mean_seniority(company)
        # iterating through all agents
        for ind_i, i in enumerate(company.keys()):
            for j in range(0, len(company[i])):
                id += 1
                if company[i][j] is not None:
                    update_agents(company, i, j, weights, months_pl, intervention)
                    fire_agent(company, i, j)

                if company[i][j] == None:
                    promote_agent(company, i, j, ind_i, weight=weights, bias = bias, threshold = threshold[ind_i], diversity_bias_scaler = diversity_bias_scaler, id = id, intervention = intervention, mean_senior = mean_senior)
           
                dat = {'id': company[i][j].id, 'gender': company[i][j].gender[0], 'age': company[i][j].age, 'seniority': company[i][j].seniority, 'seniority_pos': company[i][j].seniority_position, 'parental_leave': company[i][j].parental_leave, 'position': company[i][j].position, 'tick': month}
                adata = adata.append(dat, ignore_index=True, verify_integrity=False, sort=False)
            
        # plotting and appending data to data frame                           
        if plot_each_tick:
            plot_gender(company, tick = month)
        
        counts = count_gender(company)
        f = {'gender': 'female', 'tick': month, 'Level 1': counts[0][0],'Level 2': counts[0][1], 'Level 3': counts[0][2], 'Level 4': counts[0][3], 'Level 5': counts[0][4], 'Level 6': counts[0][5]}
        m = {'gender': 'male', 'tick': month, 'Level 1': counts[1][0],'Level 2': counts[1][1], 'Level 3': counts[1][2], 'Level 4': counts[1][3], 'Level 5': counts[1][4], 'Level 6': counts[1][5] }

        # create pandas dataframe from dictionaries f and m
        new_data = pd.DataFrame.from_dict([f, m])
        data = data.append(new_data, ignore_index=False, verify_integrity=False, sort=False)

        print('tick {} done'.format(month))

    # plotting the gender development over time
    plot_gender_development(company, months = months, data = data)
    # saving the data to a csv-file
    adata.to_csv(save_path)

    # saving the company as is at the end of the simulation
    if intervention == None:
        save_dict(company)
