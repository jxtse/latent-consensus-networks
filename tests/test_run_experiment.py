from types import SimpleNamespace

import pytest

from experiments import run_experiment
from lcn.configs import ModelConfig


class TestBuildModel:
    def test_qwen_backend_defaults_to_official_model_name(self, monkeypatch):
        captured = {}

        class StubQwenWrapper:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.hidden_dim = kwargs["hidden_dim"] or 256

        monkeypatch.setattr(run_experiment, "Qwen35ModelWrapper", StubQwenWrapper)

        model = run_experiment.build_model(
            ModelConfig(
                backend="qwen3.5",
                model_name="",
                hidden_dim=None,
                device="cpu",
                torch_dtype="auto",
            )
        )

        assert model.hidden_dim == 256
        assert captured["model_name"] == "Qwen/Qwen3.5-4B"
        assert captured["torch_dtype"] == "auto"

    def test_build_model_wrappers_assigns_one_cuda_device_per_agent(self, monkeypatch):
        captured = []

        class StubQwenWrapper:
            def __init__(self, **kwargs):
                captured.append(kwargs)
                self.hidden_dim = 256
                self.device = kwargs["device"]

        monkeypatch.setattr(run_experiment, "Qwen35ModelWrapper", StubQwenWrapper)
        monkeypatch.setattr(run_experiment.torch.cuda, "device_count", lambda: 4)

        agents = [SimpleNamespace(agent_id=0), SimpleNamespace(agent_id=1)]
        wrappers = run_experiment.build_model_wrappers(
            ModelConfig(
                backend="qwen3.5",
                model_name="Qwen/Qwen3.5-4B",
                hidden_dim=None,
                device="cuda",
                torch_dtype="bfloat16",
                per_agent_gpu=True,
            ),
            agents=agents,
        )

        assert sorted(wrappers) == [0, 1]
        assert wrappers[0].device == "cuda:0"
        assert wrappers[1].device == "cuda:1"
        assert [entry["device"] for entry in captured] == ["cuda:0", "cuda:1"]

    def test_build_model_wrappers_rejects_agent_count_exceeding_gpus(self, monkeypatch):
        monkeypatch.setattr(run_experiment.torch.cuda, "device_count", lambda: 1)

        agents = [SimpleNamespace(agent_id=0), SimpleNamespace(agent_id=1)]
        with pytest.raises(ValueError, match="one visible GPU per agent"):
            run_experiment.build_model_wrappers(
                ModelConfig(
                    backend="qwen3.5",
                    model_name="Qwen/Qwen3.5-4B",
                    device="cuda",
                    per_agent_gpu=True,
                ),
                agents=agents,
            )
