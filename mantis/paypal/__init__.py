"""PayPal authorization gate: the agent reserves funds, MANTIS decides who moves them.

Modules, in dependency order:

``config``     mock/live switch; live is the PayPal **sandbox**, by construction
``models``     Mandate, PurchaseIntent, AgentContext (scope reuses MandateScope)
``merchants``  merchant profiles (the MCC a PayPal order lacks) and the KYA registry
``client``     PayPalClient protocol, LiveClient, MockClient, AgentPayPal
``receipt``    canonical receipt -> sha256 -> the order's custom_id
``adapter``    PayPal order -> frozen TxEvent, and intent -> Orders v2 body
``gate``       PaymentGate: the only caller of capture and void
``scenarios``  the three demo scenarios, shared by the demo and the tests

Sits after ``mantis.defense`` in the dependency order and is imported only by
``mantis.api``. Nothing in ``defense`` imports it.
"""
