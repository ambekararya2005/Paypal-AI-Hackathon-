"""The AI side of the gate: a demo store, a tool-calling shopping agent, the
mandate compiler, and the D/E/F/G scenarios.

Sits after ``mantis.paypal`` in the dependency order. The agent pays only through
:class:`~mantis.paypal.client.AgentPayPal`, via the gate, and its provenance is
recorded by the tool layer — never reported by the model.
"""
