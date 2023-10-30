import numpy as np

"""
    System Setup
"""

# create 5 arms in a fixed order
arm_order = list(range(0, 200))

# save the number of arms
NUM_ARMS = len(arm_order)

"""
    Display Setup
"""

# display all floating point numbers to 3 decimal places
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

"""
    Helper Functions
"""


# return the index of the largest value in the supplied list
# arbitrarily select between the largest values in the case of a tie
# (the standard np.argmax just chooses the first value in the case of a tie)
def random_argmax(value_list):
    """ a random tie-breaking argmax"""
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


class BanditArm:
    """ the base power arm class """

    def __init__(self, q):
        self.q = q  # the true reward value
        self.initialize()  # reset the arm

    def initialize(self):
        self.Q = 0  # the estimate of this arm's reward value
        self.n = 0  # the number of times this arm has been tried

    def charge(self):
        """ return a random amount of charge """

        # the reward is a guassian distribution with unit variance around the true
        # value 'q'
        value = np.random.randn() + self.q

        # never allow a charge less than 0 to be returned
        return 0 if value < 0 else value

    def update(self, R):
        """ update this arm after it has returned reward value 'R' """

        # increment the number of times this arm has been tried
        self.n += 1

        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0 / self.n) * self.Q + (1.0 / self.n) * R

    def sample(self, t):
        """ return an estimate of the arm's reward value """
        return self.Q


class ArmTester():
    """ create and test a set of arms over a single test run """

    def __init__(self, arm=BanditArm, arm_order=arm_order, multiplier=2,
                 reward_structure='unknown', k_neighbors=20, k_drift=0, **kwargs):

        # create supplied arm type with a mean value defined by arm order
        self.arms = [arm(q, **kwargs) for q in arm_order]

        # set the number of arms equal to the number created
        self.number_of_arms = len(self.arms)

        # the index of the best arm is the last in the arm_order list
        # - this is a one-based value so convert to zero-based
        self.optimal_arm_index = (arm_order[-1] - 1)

        # by default a arm tester records 2 bits of information over a run
        self.number_of_stats = kwargs.pop('number_of_stats', 2)

        self.reward_structure = reward_structure
        self.k_neighbors = k_neighbors
        self.k_drift = k_drift

    def reset_arms(self, arm=BanditArm, arm_order=arm_order, **kwargs):
        """ reset counters at the start of a run """

        # save the number of steps over which the run will take place
        self.arms = [arm(q, **kwargs) for q in arm_order]

    def initialize_run(self, number_of_steps):
        """ reset counters at the start of a run """

        # save the number of steps over which the run will take place
        self.number_of_steps = number_of_steps

        # reset the actual number of steps that the test ran for
        self.total_steps = 0

        # monitor the total reward obtained over the run
        self.total_reward = 0

        # the current total reward at each timestep of the run
        self.total_reward_per_timestep = []

        # the actual reward obtained at each timestep
        self.reward_per_timestep = []

        # the actual reward obtained at each timestep
        self.arm_charged_per_timestep = []

        # stats for each time-step
        # - by default records: estimate, number of trials
        self.arm_stats = np.zeros(shape=(number_of_steps + 1,
                                            self.number_of_arms,
                                            self.number_of_stats))

        # ensure that all arms are re-initialized
        for arm in self.arms: arm.initialize()

    def charge_and_update(self, arm_index):
        """ charge from & update the specified arm and associated parameters """

        # charge from the chosen arm and update its mean reward value
        reward = self.arms[arm_index].charge()
        self.arms[arm_index].update(reward)

        # update the total reward
        self.total_reward += reward

        # store the current total reward at this timestep
        self.total_reward_per_timestep.append(self.total_reward)

        # store the reward obtained at this timestep
        self.reward_per_timestep.append(reward)

        #store the arm chosen at this timestep
        self.arm_charged_per_timestep.append(arm_index)

    def get_arm_stats(self, t):
        """ get the current information from each arm """
        arm_stats = [[arm.Q, arm.n] for arm in self.arms]
        return arm_stats

    def get_mean_reward(self):
        """ the total reward averaged over the number of time steps """
        return (self.total_reward / self.total_steps)

    def get_total_reward_per_timestep(self):
        """ the cumulative total reward at each timestep of the run """
        return self.total_reward_per_timestep

    def get_reward_per_timestep(self):
        """ the actual reward obtained at each timestep of the run """
        return self.reward_per_timestep

    def get_estimates(self):
        """ get the estimate of each arm's reward at each timestep of the run """
        return self.arm_stats[:, :, 0]

    def get_number_of_trials(self):
        """ get the number of trials of each arm at each timestep of the run """
        return self.arm_stats[:, :, 1]

    def get_arm_percentages(self):
        """ get the percentage of times each arm was tried over the run """
        return (self.arm_stats[:, :, 1][self.total_steps] / self.total_steps)

    def get_optimal_arm_percentage(self):
        """ get the percentage of times the optimal arm was tried """
        final_trials = self.arm_stats[:, :, 1][self.total_steps]
        return (final_trials[self.optimal_arm_index] / self.total_steps)

    def get_time_steps(self):
        """ get the number of time steps that the test ran for """
        return self.total_steps

    def select_arm(self, t):
        """ Arm Selection"""

        # Assuming prior knowledge of the concavity of reward function exists, we may will constrain Thomson Sampling to
        # choose actions within k ordinals of distance relative to the previous action.
        # In addition, and more importantly, the next action will also be constrained to happen such that concavity is
        # exploited, and we move toward directions of increased rewards and away from directions of diminishing rewards.

        if self.reward_structure=='concave' and t>1:

            try:
                exploration_upper_bound = self.arm_charged_per_timestep[t - 2] + self.k_neighbors - self.k_drift
            except:
                exploration_upper_bound = self.number_of_arms

            try:
                exploration_lower_bound = self.arm_charged_per_timestep[t - 2] - self.k_neighbors + self.k_drift
            except:
                exploration_lower_bound = 0


            if self.reward_per_timestep[t-1]>self.reward_per_timestep[t-2] \
                    and self.arm_charged_per_timestep[t-1]>=self.arm_charged_per_timestep[t-2]:

                candidate_arms=self.arms[(self.arm_charged_per_timestep[t - 2] - self.k_drift):exploration_upper_bound]

            elif self.reward_per_timestep[t-1]<self.reward_per_timestep[t-2] \
                    and self.arm_charged_per_timestep[t-1]>=self.arm_charged_per_timestep[t-2]:

                candidate_arms = self.arms[exploration_lower_bound:(self.arm_charged_per_timestep[t - 2] + self.k_drift)]

            elif self.reward_per_timestep[t-1]>self.reward_per_timestep[t-2] \
                    and self.arm_charged_per_timestep[t-1]<self.arm_charged_per_timestep[t-2]:

                candidate_arms = self.arms[exploration_lower_bound:(self.arm_charged_per_timestep[t - 2] + self.k_drift)]

            elif self.reward_per_timestep[t-1]<self.reward_per_timestep[t-2] \
                    and self.arm_charged_per_timestep[t-1]<self.arm_charged_per_timestep[t-2]:

                candidate_arms = self.arms[(self.arm_charged_per_timestep[t - 2] - self.k_drift):exploration_upper_bound]

            elif self.arm_charged_per_timestep[t-1]==self.arm_charged_per_timestep[t-2]:

                candidate_arms = self.arms[(self.arm_charged_per_timestep[t - 2] - self.k_drift // 2):
                                                 (self.arm_charged_per_timestep[t - 2] + self.k_drift // 2)]

            arm_index = random_argmax([arm.sample(t + 1) if np.isin(arm, candidate_arms) else -np.inf for arm in self.arms])

        else:

            # choose the arm with the current highest mean reward or arbitrarily
            arm_index = random_argmax([arm.sample(t + 1) for arm in self.arms])

        return arm_index

    def run(self, number_of_steps, maximum_total_reward=float('inf')):
        """ perform a single run, over the set of arms,
            for the defined number of steps """

        # reset the run counters
        self.initialize_run(number_of_steps)

        # loop for the specified number of time-steps
        for t in range(number_of_steps):

            # print(t); print(self.arm_charged_per_timestep); print(self.reward_per_timestep); time.sleep(0.1)

            # get information about all arms at the start of the time step
            self.arm_stats[t] = self.get_arm_stats(t)

            # select a arm
            arm_index = self.select_arm(t)

            # charge from the chosen arm and update its mean reward value
            self.charge_and_update(arm_index)

            # test if the accumulated total reward is greater than the maximum
            if self.total_reward > maximum_total_reward:
                break

        # save the actual number of steps that have been run
        self.total_steps = t

        # get the stats for each arm at the end of the run
        self.arm_stats[t + 1] = self.get_arm_stats(t + 1)

        return self.total_steps, self.total_reward


class ArmExperiment():
    """ setup and run repeated arm tests to get the average results """

    def __init__(self,
                 arm_tester=ArmTester,
                 number_of_tests=300,
                 number_of_steps=50,
                 maximum_total_reward=float('inf'),
                 **kwargs):

        self.arm_tester = arm_tester
        self.number_of_tests = number_of_tests
        self.number_of_steps = number_of_steps
        self.maximum_total_reward = maximum_total_reward
        self.number_of_arms = self.arm_tester.number_of_arms

    def initialize_run(self):

        # keep track of the average values over the run
        self.mean_total_reward = 0.
        self.optimal_selected = 0.
        self.mean_time_steps = 0.
        self.arm_percentages = np.zeros(self.number_of_arms)
        self.estimates = np.zeros(shape=(self.number_of_steps + 1, self.number_of_arms))
        self.number_of_trials = np.zeros(shape=(self.number_of_steps + 1, self.number_of_arms))

        # the cumulative total reward per timestep
        self.cumulative_reward_per_timestep = np.zeros(shape=(self.number_of_steps))

        # the actual reward obtained at each timestep
        self.reward_per_timestep = np.zeros(shape=(self.number_of_steps))

    def get_mean_total_reward(self):
        """ the final total reward averaged over the number of timesteps """
        return self.mean_total_reward

    def get_cumulative_reward_per_timestep(self):
        """ the cumulative total reward per timestep """
        return self.cumulative_reward_per_timestep

    def get_reward_per_timestep(self):
        """ the mean actual reward obtained at each timestep """
        return self.reward_per_timestep

    def get_optimal_selected(self):
        """ the mean times the optimal arm was selected """
        return self.optimal_selected

    def get_arm_percentages(self):
        """ the mean of the percentage times each arm was selected """
        return self.arm_percentages

    def get_estimates(self):
        """ per arm reward estimates """
        return self.estimates

    def get_number_of_trials(self):
        """ per arm number of trials """
        return self.number_of_trials

    def get_mean_time_steps(self):
        """ the average number of trials of each test """
        return self.mean_time_steps

    def update_mean(self, current_mean, new_value, n):
        """ calculate the new mean from the previous mean and the new value """
        return (1 - 1.0 / n) * current_mean + (1.0 / n) * new_value

    def update_mean_array(self, current_mean, new_value, n):
        """ calculate the new mean from the previous mean and the new value for an array """

        new_value = np.array(new_value)

        # pad the new array with its last value to make sure its the same length as the original
        pad_length = (current_mean.shape[0] - new_value.shape[0])

        if pad_length > 0:
            new_array = np.pad(new_value, (0, pad_length), mode='constant', constant_values=new_value[-1])
        else:
            new_array = new_value

        return (1 - 1.0 / n) * current_mean + (1.0 / n) * new_array

    def record_test_stats(self, n):
        """ update the mean value for each statistic being tracked over a run """

        # calculate the new means from the old means and the new value
        tester = self.arm_tester
        self.mean_total_reward = self.update_mean(self.mean_total_reward, tester.get_mean_reward(), n)
        self.optimal_selected = self.update_mean(self.optimal_selected, tester.get_optimal_arm_percentage(), n)
        self.arm_percentages = self.update_mean(self.arm_percentages, tester.get_arm_percentages(), n)
        self.mean_time_steps = self.update_mean(self.mean_time_steps, tester.get_time_steps(), n)

        self.cumulative_reward_per_timestep = self.update_mean_array(self.cumulative_reward_per_timestep,
                                                                     tester.get_total_reward_per_timestep(), n)

        # check if the tests are only running until a maximum reward value is reached
        if self.maximum_total_reward == float('inf'):
            self.estimates = self.update_mean_array(self.estimates, tester.get_estimates(), n)
            self.cumulative_reward_per_timestep = self.update_mean_array(self.cumulative_reward_per_timestep,
                                                                         tester.get_total_reward_per_timestep(), n)
            self.reward_per_timestep = self.update_mean_array(self.reward_per_timestep,
                                                              tester.get_reward_per_timestep(), n)
            self.number_of_trials = self.update_mean_array(self.number_of_trials, tester.get_number_of_trials(), n)

    def run(self):
        """ repeat the test over a set of arms for the specified number of trials """

        # do the specified number of runs for a single test
        self.initialize_run()
        for n in range(1, self.number_of_tests + 1):
            # do one run of the test
            self.arm_tester.run(self.number_of_steps, self.maximum_total_reward)
            self.record_test_stats(n)
            for arm in self.arm_tester.arms: arm.__init__(arm.q)
