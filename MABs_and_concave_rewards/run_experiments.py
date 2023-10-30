# # A Comparison of Bandit Algorithms

from MABs_and_concave_rewards.BanditExperimentation import *


# ## The Optimistic Greedy Power Arm
# Create an Optimistic Arm class by inheriting from the standard Bandit Arm

class OptimisticArm( BanditArm ):
    def __init__( self, q, **kwargs ):    
                      
        # get the initial reqrd estimate from the kwargs
        self.initial_estimate = kwargs.pop('initial_estimate', 0.) 
        
        # pass the true reward value to the base PowerArm             
        super().__init__(q)         
                
    def initialize(self):        
        # estimate of this arm's reward value 
        # - set to supplied initial value
        self.Q = self.initial_estimate    
        self.n = 0    


# ## The Upper Confidence Bounds Arm

class UCBArm( BanditArm ):

    def __init__( self, q, **kwargs ):    
        """ initialize the UCB arm """                  
        
        # store the confidence level controlling exploration
        self.confidence_level = kwargs.pop('confidence_level', 2.0)        
                
        # pass the true reward value to the base PowerArm   
        super().__init__(q)           
        
    def uncertainty(self, t): 
        """ calculate the uncertainty in the estimate of this arm's mean """
        if self.n == 0: return float('inf')                         
        return self.confidence_level * (np.sqrt(np.log(t) / self.n))         
        
    def sample(self,t):
        """ the UCB reward is the estimate of the mean reward plus its uncertainty """
        return self.Q + self.uncertainty(t) 


# ## The Gaussian Thompson Sampling Arm

class GaussianThompsonArm( BanditArm ):
    def __init__(self, q):                
                
        self.τ_0 = 0.0001  # the posterior precision
        self.μ_0 = 1       # the posterior mean
        
        # pass the true reward value to the base PowerArm             
        super().__init__(q)         
        
    def sample(self,t):
        """ return a value from the the posterior normal distribution """
        return (np.random.randn() / np.sqrt(self.τ_0)) + self.μ_0    
                    
    def update(self,R):
        """ update this arm after it has returned reward value 'R' """   

        # do a standard update of the estimated mean
        super().update(R)    
               
        # update the mean and precision of the posterior
        self.μ_0 = ((self.τ_0 * self.μ_0) + (self.n * self.Q))/(self.τ_0 + self.n)        
        self.τ_0 += 1       


# The Epsilon Greedy Arm Tester
# 
# Note that Epsilon Greedy just uses the standard power arm. 
# 
# Instead of cusomizing the power arm class it instead modifies the arm selection algorithm, to randomly select from the complete set of arms when the probability value is less than the defined value of epsilon.
# 
# All other algorithms just use the standard arm selection routine, which always chooses the arm that returns the highest reward on the current time-step.


class EpsilonGreedyArmTester( ArmTester ):

    def __init__(self, arm_order=arm_order, multiplier=2, epsilon = 0.2 ):  
        
        # create a standard arm tester
        super().__init__(arm_order=arm_order, multiplier=multiplier) 
        
        # save the probability of selecting the non-greedy action
        self.epsilon = epsilon
    
    
    def select_arm( self, t ):
        """ Epsilon-Greedy arm Selection"""
        
        # probability of selecting a random arm
        p = np.random.random()

        # if the probability is less than epsilon then a random arm is chosen from the complete set
        if p < self.epsilon:
            arm_index = np.random.choice(self.number_of_arms)
        else:
            # choose the arm with the current highest mean reward or arbitrary select a arm in the case of a tie            
            arm_index = random_argmax([arm.sample(t) for arm in self.arms])               
        
        return arm_index


# Testing on the standard power arm problem


methods = []
rewards = []
mean_rewards = []

def run_multiple_tests( tester, max_steps = 500, show_arm_percentages = True ):
    number_of_tests = 100
    number_of_steps = max_steps
    maximum_total_reward = 3600

    experiment = ArmExperiment(arm_tester   = tester,
                                  number_of_tests = number_of_tests,
                                  number_of_steps = number_of_steps,
                                  maximum_total_reward = maximum_total_reward)
    experiment.run()

    print(f'Mean Reward per Time Step = {experiment.get_mean_total_reward():0.3f}')
    print(f'Optimal Arm Selected = {experiment.get_optimal_selected():0.3f}')    
    print(f'Average Number of Trials Per Run = {experiment.get_mean_time_steps():0.3f}')
    if show_arm_percentages:
        print(f'Arm Percentages = {experiment.get_arm_percentages()}') 
        
    rewards.append(experiment.get_cumulative_reward_per_timestep())        
    mean_rewards.append(f"{experiment.get_mean_total_reward():0.3f}")


# Greedy Selection
run_multiple_tests( ArmTester( BanditArm ) )

# Epsilon Greedy
run_multiple_tests( EpsilonGreedyArmTester( epsilon = 0.2 ) )

# Optimistic Greedy
run_multiple_tests( ArmTester( OptimisticArm, initial_estimate = 20. ))

# Upper Confidence Bound (derived from Hoeffding's inequality)
run_multiple_tests( ArmTester( UCBArm, confidence_level = 0.6 ))

# Thompson Sampling
run_multiple_tests( ArmTester( GaussianThompsonArm ))

# Thompson Sampling with Concavity Constrains (satisfactory results, though there's potential for improvement in
# updating unused arms according to the expected improvement directionality.
# Also, code can be made nicer via reparameterization of the drift. And k-neighbor proportional to the number of arms.
run_multiple_tests( ArmTester( GaussianThompsonArm, reward_structure='concave' ))

# To explore further Convex Bandit techniques as well as Stochastic Approximation methods.
