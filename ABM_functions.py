import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# FUNCTIONS FOR OUR ABM

# defining agents as abstract class.
class Agent:
    # init-method, the constructor method for agents
    def __init__(self, age, index, attachment_category, symptom_score, symptom_level, initial_symptom_score, concussion_level,  start_energy_level, end_energy_level, recovery_slope, recovery_midpoint, seniority, id):
        self.age = age
        self.index = index
        self.attachment_category = attachment_category
        self.symptom_score = symptom_score
        self.symptom_level = symptom_level
        self.initial_symptom_score = initial_symptom_score
        self.start_energy_level = start_energy_level
        self.end_energy_level = end_energy_level
        self.concussion_level = concussion_level
        self.recovery_slope = recovery_slope
        self.recovery_midpoint = recovery_midpoint
        self.seniority = seniority
        self.id = id

# function for creating empty dictionary
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

def get_truncated_normal(mean=0, sd = 1, lower = 20, upper = 67, size=1):
    '''
    Draws a random sample from a truncated normal distribution. Used for computing the age of each agent in the populate_labourmarket function
    '''
    samples = np.random.normal(mean, sd, size)
    truncated_samples = np.clip(samples, lower, upper)
    return truncated_samples

def get_symptom_score(weights: dict):
    '''
    Computes the initial symptom score for each agent by randomly sampling a value between 0-6, corresponding to the PCSS likert scale, for each symptom

    Parameters
    ----------
    weights: dictionary with information about the symptoms
    '''
    symptoms = []
    for i in range(len(weights['symptoms'])):    # iterating over symptom dictionary names and their values
        symptom = np.random.poisson(3, size=1)  # poisson distribution, as symptom score only can take positive integers between 0 and 6
        symptom = np.clip(symptom, 0, 6) # bounding at max = 6, so that scale corresponds to PCSS
        symptoms.append(symptom[0])
    initial_symptom_score = sum(symptoms)
    return initial_symptom_score

def get_symptom_level(initial_symptom_score):
    '''
    Computes the symptom level (is later used for plotting different graphs for severity)
    Thresholds corresponds to the four quantiles of initial_symptom_score.

    Parameters
    ----------
    initial_symptom_score: integer
    '''
    if initial_symptom_score > 42:
        symptom_level = 4
    elif initial_symptom_score >= 38:
        symptom_level = 3
    elif initial_symptom_score > 34:
        symptom_level = 2
    elif initial_symptom_score <= 34:
        symptom_level = 1
    
    return symptom_level

def get_recovery_slope():
    '''
    Assigns a recovery slope to each agent from one of the following distributions, later used to compute individual recovery rates
    '''
    probabilities = [0.65, 0.35]
    random_choice = random.choices([0, 1], probabilities)[0]
    if random_choice == 0:
        recovery_slope = random.gauss(0.75, 0.2)
    if random_choice == 1:
        recovery_slope = random.gauss(0.03, 0.06)

    return recovery_slope

def get_concussion_level(tick, symptom_score):
    '''
    Calculates the concussion level based on symptom score. Is not calculated until week 12 to allign with real data

    Parameters
    ----------
    Tick: integer, weeks
    Symptom_score: symptom score of the agent
    '''
    if  tick <= 12 and symptom_score <= 1:
        concussion_level = 'mild'
    else:
        concussion_level = 'pcs'
    return concussion_level

def populate_labourmarket(labourmarket: dict, weights: dict):
    '''
    Creates the first agents and puts them in the labourmarket dictionary.

    Parameters
    ----------
    labourmarket : dictionary, a dictionary with the attachment as keys and empty lists as values
    weights: dictionary, a dictionary containing information on how to generate symptom_score of the agents in each attachment
    '''
    id = 0 # no agents in beginning
    for i in labourmarket.keys(): # iterating over all categories of labour market
        for j in range(0, len(labourmarket[i])): # iterating over only first attachment category: we only want to have normal job agents
            id += 1 # creating one id (= one agent)
            initial_symptom_score = get_symptom_score(weights) # has to be defined here, so that outcome can be used for computing concussion level and energy_level inside agent
            labourmarket[i][j] = Agent(
                attachment_category = i, 
                index = j,
                age = get_truncated_normal(44, 10), # 44 and 10, based on average age of peopple attached to the danish labour market
                symptom_score = 0,
                initial_symptom_score = initial_symptom_score,
                symptom_level = get_symptom_level(initial_symptom_score),
                concussion_level = 0, # 0 is initial value, as it is specified at tick 12
                start_energy_level = random.gauss(90,10), # arbitrary value used for defining when an agent should be in a which attachment group
                end_energy_level = 0, # is updated each tick
                recovery_slope = get_recovery_slope(),
                recovery_midpoint = random.gauss(2,1), # recovery trajectory is computed as a sigmoid curve, this value gives each agent an individual midpoint
                seniority = 0, # counts how long an agent has been in the same attachment category. Used to decide when an agent should change attachment group
                id = id)
        break
    return id

def count_level_agents(data: pd.DataFrame, percentage: bool):
    """
    Counts the number of mild and PCS concussions in each attachment category for each tick.
    _____________
    Parameters:
    data: DataFrame containing agent data.
    percentage: if True, results are shown in percentage, if False, results are shown in real numbers
    """
    # Group by tick, concussion_level, and attachment_category and count occurrences
    counts = data.groupby(['tick', 'concussion_level', 'attachment_category']).size().unstack(fill_value=0)

    # Ensure that all attachment categories are included in the returned dataframe
    all_attachment_categories = ['Normal Job', 'Sick Leave', 'Retired', 'Disability Pension', 'Flex Job', 'Job Clarification']
    counts = counts.reindex(columns=all_attachment_categories, fill_value=0).reset_index()
    
    if percentage == True:
        combined_counts = counts.groupby('tick').sum().reset_index()

        # Calculate the total number of agents for each tick
        combined_counts['total'] = combined_counts[all_attachment_categories].sum(axis=1)

        # Merge combined counts back with original counts
        counts = pd.merge(counts, combined_counts[['tick', 'total']], on='tick', how='left')

        # Calculate the percentage for each attachment category and concussion level
        for category in all_attachment_categories:
            counts[category] = (counts[category] / counts['total']) * 100

    return counts

def update_agents(labourmarket, i, j, tick, intervention = None):
    '''
    Updates the agents each tick, eg: age, energy_level and symptom_score

    Parameters
    ----------
    Labourmarket : dictionary
    i : Attachment category
    j : the index of the agent at each attachment category
    tick: week
    intervention: whether simulation is with intervention or not, automatically sat to none
    '''
    # other recovery slopes for intervention and if recovery slope is negative at tick 3
    if tick == 3 and labourmarket[i][j].recovery_slope < 0.045:
        if labourmarket[i][j].recovery_slope < 0:
            labourmarket[i][j].recovery_slope = 0
        if intervention == 'intervention':
            labourmarket[i][j].recovery_slope = random.gauss(0.075, 0.025)
    
    # updates concussion_level (is done at week 12)
    if tick == 12:
        symptom_score = labourmarket[i][j].symptom_score
        labourmarket[i][j].concussion_level = get_concussion_level(tick, symptom_score)
    
    # update age:
    labourmarket[i][j].age = labourmarket[i][j].age + 1/52 

    #updating symptom_score: recovery rate follows a sigmoid curve with individual slope and midpoint parameters for each agent
    labourmarket[i][j].symptom_score = labourmarket[i][j].initial_symptom_score * (1 / (1 + np.exp(labourmarket[i][j].recovery_slope * (tick - labourmarket[i][j].recovery_midpoint))))
    if labourmarket[i][j].symptom_score < 0:
        labourmarket[i][j].symptom_score = 0

    # updating energy_level
    labourmarket[i][j].end_energy_level = labourmarket[i][j].start_energy_level - labourmarket[i][j].symptom_score 

# function for moving each agent
def move (new, labourmarket: dict, i, j):
    '''
    Function for moving the agents from one attachment category to another.

    Parameters
    ----------
    New: attachment category the agent is moved to
    Labourmarket : dictionary
    i : Attachment category
    j : the index of the agent at each attachment category
    '''
    agent = labourmarket[i][j]
    new_group = new
    labourmarket[new][j] = agent
    if new_group != i:
        labourmarket[i][j] = None
        labourmarket[new][j].seniority = 0
    elif new_group == i:
        labourmarket[new][j].seniority = labourmarket[new][j].seniority + 1


def move_agents (labourmarket: dict, i, j, tick):
    '''
    Rules for when an agent is moved. Moves the agents from one attachment category to another.

    Parameters
    ----------
    Labourmarket : dictionary
    i : Attachment category
    j : the index of the agent at each attachment category
    tick : week
    '''
    if labourmarket[i][j] is not None:
        
        if labourmarket[i][j].age >= 67:
            move('Retired', labourmarket, i, j)

        elif labourmarket[i][j].end_energy_level >= 65:
            move('Normal Job', labourmarket, i, j)
        
        elif labourmarket[i][j].end_energy_level < 65 and tick <= 22:
            move('Sick Leave', labourmarket, i, j)
        
        elif i == 'Flex Job':
            move('Flex Job', labourmarket, i, j)
        
        elif i == 'Disability Pension':
            move('Disability Pension', labourmarket, i, j)

        elif labourmarket[i][j].end_energy_level < 65 and labourmarket[i][j].seniority <= 104: # only allowed to be in job activation for maksimum of 2 years
            dp = random.gauss(23, 9) # number that influences how long it takes to be assigned a disability pension
            job = random.gauss(76, 24) # number that influences how long it takes to come back to a job again: 1.5 years in weeks = 76, 0.5 years in weeks = 24
            if labourmarket[i][j].end_energy_level < 62 and labourmarket[i][j].recovery_slope < 0.003 and dp < labourmarket[i][j].seniority: # if agents recovery slope is almost constant and seniority in job clarification is greater than the dp number, agent is assigned a flex job
                move('Disability Pension', labourmarket, i, j)
            elif job < labourmarket[i][j].seniority: # if agent seniority in job clarification is greater than the job number, agent is assigned a flex job
                move('Flex Job', labourmarket, i, j)
            elif job > labourmarket[i][j].seniority:
                move('Job Clarification', labourmarket, i, j)
            else:
                move('Job Clarification', labourmarket, i, j)
        

### PLOTTING ###
def plot_levels(labourmarket, tick):
    '''
    Plots the current level distribution in each attachment category of the labourmarket

    Parameters
    ---------- 
    labourmarket : dictionary, a dictionary with the attachment groups as keys and lists of agents as values
    tick : the tick that should be plotted
    '''
    mild, pcs = count_level_agents(labourmarket)
    
    groups = labourmarket.keys()
    x = np.arange(len(groups)) 
    width = 0.2

    fig, ax = plt.subplots(figsize=(9,6))
    rects1 = ax.bar(x - width, mild, width, label = 'Mild', color = 'skyblue')
    rects2 = ax.bar(x, pcs, width, label = 'Pcs', color = 'steelblue')

    ax.set_ylabel('Count')
    ax.set_title('Count by Attachment Category and Level of Concussion, week = {}'.format(tick + 1))
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    return ax

def plot_levels_development(data: pd.DataFrame, percentage = False):
    '''
    Plots the current level distribution in each attachment category of the labourmarket

    Parameters
    ---------- 
    data: dataframe with data from all agents for each tick
    percentage: bool, if True, output displays percentage, if False, output displays real numbers
    '''

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharey=False)
    colorlist = {'mild': 'lightblue', 'pcs': 'darkslategray'}
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colorlist.values()]
    plt.legend(markers, colorlist.keys(), numpoints=1) 
    fig.suptitle('Counts of each concussion level in each attachment group', fontsize=16)
    fig.text(0.5, 0.01, 'Week', ha='center', va='center') 
    fig.text(0.01, 0.5, 'Count', ha='center', va='center', rotation='vertical')

    if percentage == True: 
        for index, attachment in enumerate(data.columns.drop(['concussion_level', 'tick', 'total'])):
            ax = axs.flatten()[index]
            ax.set_title(attachment)

            for level in ['mild', 'pcs']:
                level_dat = data.loc[data['concussion_level'] == level]
                dat = level_dat[attachment]
                ticks = level_dat['tick']
                ax.plot(ticks, dat, color=colorlist[level])
    else:
        for index, attachment in enumerate(data.columns.drop(['concussion_level', 'tick'])):
            ax = axs.flatten()[index]
            ax.set_title(attachment) 

            for level in ['mild', 'pcs']:
                level_dat = data.loc[data['concussion_level'] == level]
                dat = level_dat[attachment]
                ticks = level_dat['tick']
                ax.plot(ticks, dat, color=colorlist[level])

    plt.tight_layout()
    plt.show()

def compute_confidence_intervals(data: pd.DataFrame):
    '''
    Computes symptom score summary statistics (mean, ci) for each concussion level for each tick

    Parameters
    ---------- 
    data: dataframe with information from all agents for each tick
    '''
    pd.to_numeric(data['tick'])
    pd.to_numeric(data['symptom_score'])

    ci_data = data.groupby(['concussion_level', 'symptom_level', 'tick'])['symptom_score'].agg(['mean', 'std']).reset_index()
    
    ci_data['l'] = ci_data['mean'] + 1.96 * ci_data['std']
    ci_data['u'] = ci_data['mean'] - 1.96 * ci_data['std']

    return ci_data

def plot_symptom_score_development(data: pd.DataFrame):
    '''
    Plots the average development in symptom_score over time (ticks)

    Parameters
    ---------- 
    data: dataframe with symptom score summary statistics for each concussion level and symptom level from each tick
    '''
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), sharey=True)
    colorlist = {
        ('mild', 1): 'lightsteelblue', ('mild', 2): 'slategray', ('mild', 3): 'royalblue', ('mild', 4): 'darkblue',
        ('pcs', 1): 'lightgreen', ('pcs', 2): 'darkseagreen', ('pcs', 3): 'olive', ('pcs', 4): 'darkolivegreen'
    }
    
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colorlist.values()]
    labels = [f'{concussion.capitalize()} - Level {symptom}' for concussion, symptom in colorlist.keys()]
    fig.legend(markers, labels, numpoints=1, loc='upper right', fontsize='small')
    
    fig.suptitle('Development in Symptom Score', fontsize=16)
    fig.text(0.5, 0.01, 'Week', ha='center', va='center')
    fig.text(0.01, 0.5, 'Average Symptom Score', ha='center', va='center', rotation='vertical')
    
    for index, level in enumerate(['mild', 'pcs']):
        ax = axs.flatten()[index]
        ax.set_title(level.capitalize())

        for symptom_level in [1, 2, 3, 4]:
            level_data = data[(data['concussion_level'] == level) & (data['symptom_level'] == symptom_level)]
            color = colorlist[(level, symptom_level)]
            
            # Mirroring the y-values along the x-axis (to make visually comparable to calibration data)
            mirrored_mean = level_data['mean'] * -1
            mirrored_l = level_data['l'] * -1
            mirrored_u = level_data['u'] * -1
            
            # Clipping the upper bound at 0 to ensure no positive values
            mirrored_u = np.clip(mirrored_u, None, 0)
            
            ax.plot(level_data['tick'], mirrored_mean, color=color, label=f'Level {symptom_level} Mean', linewidth=2)
            ax.fill_between(level_data['tick'], mirrored_l, mirrored_u, color=color, alpha=0.3, label=f'Level {symptom_level} 95% CI')

        # Add horizontal line at y = 0
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def update_concussion_level(data: pd.DataFrame):
    '''
    Updates the concussion level of each agent for all ticks (also prior, as concussion level is not defined before tick 12, it is sat to 0 before that)

    Parameters
    ---------- 
    data: dataframe with information from all agents for each tick
    '''
    unique_ids = data['id'].unique()
    for id in unique_ids:
        agent_rows = data[data['id'] == id]
        if 'mild' in agent_rows['concussion_level'].values:
            data.loc[data['id'] == id, 'concussion_level'] = 'mild'
        if 'pcs' in agent_rows['concussion_level'].values:
            data.loc[data['id'] == id, 'concussion_level'] = 'pcs'
    return data

# Function for running the ABM
def run_abm(run_id, weeks: int, attachment: list, n: list, weights: dict, plot_each_tick = False, intervention = None, labourmarket = None, percentage = False):
    '''
    Runs the ABM simulation

    Parameters
    ----------
    weeks : int, the number of weeks to simulate
    attachment: list, a list of strings with the attachment categories
    n : list, a list of integers with the number of positions in each attachment group
    plot_each_tick : bool, if True, the concussion level distribution is plotted after each tick
    weights : dictionary, a dictionary containing the information used to generate agents
    intervention : if None, updated recovery trajectories are not implemented
    percentage : bool, if True, outcome is shown in percentage, if False, outcome is shown in real numbers
    '''
    # creating empty dataframe for the results
    adata = pd.DataFrame()
    
    # create labourmarket
    if labourmarket == None:
        labourmarket = create_labourmarket(attachment, n)
        # populate labourmarket using weights, and saving the last ID given to an agent
        id = populate_labourmarket(labourmarket, weights)
    else: 
        id = 0

    # plots the initial distribution
    if plot_each_tick:
        plot_levels(labourmarket, tick=-1)

    # iterating though the months
    for week in range(weeks):  
        # iterating through all agents
        for ind_i, i in enumerate(labourmarket.keys()):
            for j in range(0, len(labourmarket[i])):
                id += 1
                if week == 0 and labourmarket[i][j] is not None: # making sure agents are not updated at initial tick
                        dat = {'id': labourmarket[i][j].id, 'age': labourmarket[i][j].age, 'attachment_category': i, 'symptom_score': labourmarket[i][j].symptom_score, 'initial_symptom_score': labourmarket[i][j].initial_symptom_score, 'symptom_level': labourmarket[i][j].symptom_level, 'concussion_level': labourmarket[i][j].concussion_level, 'end_energy_level': labourmarket[i][j].end_energy_level, 'start_energy_level': labourmarket[i][j].start_energy_level, 'recovery_slope': labourmarket[i][j].recovery_slope, 'seniority': labourmarket[i][j].seniority, 'tick': week}
                        adata = adata.append(dat, ignore_index=True, verify_integrity=False, sort=False)                 
                elif labourmarket[i][j] is not None:
                    update_agents(labourmarket, i, j, week, intervention = intervention)
                    move_agents(labourmarket, i, j, week)
                    if labourmarket[i][j] is not None:
                        dat = {'id': labourmarket[i][j].id, 'age': labourmarket[i][j].age, 'attachment_category': i, 'symptom_score': labourmarket[i][j].symptom_score, 'initial_symptom_score': labourmarket[i][j].initial_symptom_score, 'symptom_level': labourmarket[i][j].symptom_level, 'concussion_level': labourmarket[i][j].concussion_level, 'end_energy_level': labourmarket[i][j].end_energy_level, 'start_energy_level': labourmarket[i][j].start_energy_level, 'recovery_slope': labourmarket[i][j].recovery_slope, 'seniority': labourmarket[i][j].seniority, 'tick': week}
                        adata = adata.append(dat, ignore_index=True, verify_integrity=False, sort=False) 

        # plots the distribution at each tick                          
        if plot_each_tick:
            plot_levels(labourmarket, tick = week)
        
        print('tick {} done'.format(week))
    
    # updates concussion level for each agent
    adata = update_concussion_level(adata)

    # counts number of agents at each tick
    count_data = count_level_agents(adata, percentage)

    # df for calculating symptom scores
    symptom_data = compute_confidence_intervals(adata)
        
    # plotting development in attachment categories and symptom score
    plot_levels_development(count_data, percentage)
    plot_symptom_score_development(symptom_data)

    if intervention == None:
        # saving the data to a csv-file
        count_data.to_csv(f'data/attachment_{run_id}.csv')
        adata.to_csv(f'data/agents_{run_id}.csv')
        symptom_data.to_csv(f'data/symptom_score_{run_id}.csv')
    else:
        count_data.to_csv(f'data/attachment_i_{run_id}.csv')
        adata.to_csv(f'data/agents_i_{run_id}.csv')
        symptom_data.to_csv(f'data/symptom_score_i_{run_id}.csv')
