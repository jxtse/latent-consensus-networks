from lcn.environments import (
    AschConformityEnvironment,
    HiddenProfileEnvironment,
    WisdomOfCrowdsEnvironment,
)


class TestHiddenProfileEnvironment:
    def test_setup_episode_returns_agents_and_task(self):
        environment = HiddenProfileEnvironment(hidden_dim=16)

        agents, task = environment.setup_episode()

        assert len(agents) == 4
        assert task["correct_option"] == "C"
        assert "options" in task


class TestAschConformityEnvironment:
    def test_setup_episode_marks_target_agent(self):
        environment = AschConformityEnvironment(hidden_dim=16)

        agents, task = environment.setup_episode()

        assert agents[0].metadata["role"] == "target"
        assert task["pressure_option"] == "A"


class TestWisdomOfCrowdsEnvironment:
    def test_setup_episode_returns_numeric_options(self):
        environment = WisdomOfCrowdsEnvironment(hidden_dim=16)

        agents, task = environment.setup_episode()

        assert len(agents) == 6
        assert task["correct_option"] == "150"
