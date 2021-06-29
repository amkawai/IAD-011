import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Manager:
    def __init__(self, env_info, agent_info, true_values_file=None, experiment_name=None):
        self.experiment_name = experiment_name
        
        self.grid_h, self.grid_w = env_info["grid_height"], env_info["grid_width"]
        self.cmap = matplotlib.cm.viridis
        self.cmap.set_bad('black', 1.)
            
        self.values_table = None
        self.policy = agent_info["policy"]
        
        self.true_values_file = true_values_file
    
    def compute_values_table(self, values):
        self.values_table = np.empty((self.grid_h, self.grid_w))
        self.values_table.fill(np.nan)
        for state in range(len(values)):
            self.values_table[np.unravel_index(state, (self.grid_h, self.grid_w))] = values[state]
                    
    def compute_RMSVE(self):
        return (np.sqrt(np.nanmean((self.values_table - self.true_values) ** 2)))
    
    def visualize(self, values, episode_num):
        if not hasattr(self, "fig"):
            self.fig = plt.figure(figsize=(10, 20))
            plt.ion()
            
            if self.true_values_file is not None:
                self.cmap_VE = matplotlib.cm.Reds
                self.cmap_VE.set_bad('black', 1.)
                self.ax = self.fig.add_subplot(311)
                self.RMSVE_LOG = []
                self.true_values = np.load(self.true_values_file)
            else:
                self.true_values = None

        self.fig.clear()
        if self.true_values is not None:
            plt.subplot(311)
        self.compute_values_table(values)
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(self.values_table, cmap=self.cmap, interpolation='nearest', origin='upper')
        
        for state in range(self.policy.shape[0]):
            for action in range(self.policy.shape[1]):
                y, x = np.unravel_index(state, (self.grid_h, self.grid_w))
                pi = self.policy[state][action]
                if pi == 0:
                    continue
                if action == 0:
                    plt.arrow(x, y, 0,  -0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 1: 
                    plt.arrow(x, y, -0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 2:
                    plt.arrow(x, y, 0, 0.5 * pi, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
                if action == 3:
                    plt.arrow(x, y, 0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1, 
                              alpha=0.5)
        
        plt.title((("" or self.experiment_name) + "\n") + "Predicted Values, Episode: %d" % episode_num)
        plt.colorbar(im, orientation='horizontal')
        
        if self.true_values is not None:
            plt.subplot(312)
            plt.xticks([])
            plt.yticks([])
            im = plt.imshow((self.values_table - self.true_values) ** 2, origin='upper', cmap=self.cmap_VE)
            plt.title('Squared Value Error: $(v_{\pi}(S) - \hat{v}(S))^2$')
            plt.colorbar(im, orientation='horizontal')
            self.RMSVE_LOG.append((episode_num, self.compute_RMSVE()))
            
            plt.subplot(313)
            plt.plot([x[0] for x in self.RMSVE_LOG], [x[1] for x in self.RMSVE_LOG])
            plt.xlabel("Episode")
            plt.ylabel("RMSVE", rotation=0, labelpad=20)
            plt.title("Root Mean Squared Value Error")
        self.fig.canvas.draw()
    
    def run_tests(self, values, RMSVE_threshold):
        assert self.true_values is not None, "This function can only be called once the true values are given during " +\
               "runtime."
        self.compute_values_table(values)
        mask = ~(np.isnan(self.values_table) | np.isnan(self.true_values))
        if self.compute_RMSVE() < RMSVE_threshold and np.allclose(self.true_values[mask], self.values_table[mask]):
            pass
        else:
            assert False
            
    def __del__(self):
        pass #plt.close()


"""Abstract environment base class for RL-Glue-py.
"""

from abc import ABCMeta, abstractmethod


class BaseEnvironment:
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    @abstractmethod
    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

    @abstractmethod
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

    @abstractmethod
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

    @abstractmethod
    def env_cleanup(self):
        """Cleanup done after the environment ends"""

    @abstractmethod
    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """


"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from abc import ABCMeta, abstractmethod

class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """


"""Glues together an experiment, agent, and environment.
"""
class RLGlue:
    """RLGlue class

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def rl_init(self, agent_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self, agent_start_info={}, env_start_info={}):
        """Starts RLGlue experiment

        Returns:
            tuple: (state, action)
        """

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        (reward, last_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward;

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
