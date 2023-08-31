Training Setup
==============

.. _InternLM-args:

Argument Parsing
----------------
InternLM uses the `argparse <https://docs.python.org/3/library/argparse.html>`_ library to supply commandline
configuration to the InternLM runtime. Use ``internlm.initialize.get_default_parser()`` to get InternLM's default
parser with some builtin arguments, users can add custom parameters to this parser.

.. code-block:: python

    # Get InternLM default parser
    parser = internlm.initialize.get_default_parser()
    # Add new argument
    parser.add_argument("--user_arg", type=int, default=-1, help="arguments add by user.")
    cmd_args = parser.parse_args()

.. autofunction:: internlm.initialize.get_default_parser


.. _InternLM-model-init:

Model Initialization
-------------------------

.. autofunction:: internlm.train.initialize_model

.. _InternLM-optim-init:

Optimizer Initialization
-------------------------

.. autofunction:: internlm.train.initialize_optimizer

.. _InternLM-dl-init:

Dataloader Initialization
-------------------------

.. autofunction:: internlm.train.get_train_data_loader

.. _InternLM-trainer-init:

Trainer Initialization
-------------------------

.. autofunction:: internlm.initialize.initialize_trainer