class WorkflowError(Exception):
    """Base class for exceptions in workflows."""
    pass


class PartialWorkflowError(WorkflowError):
    """A workflow completed some independent jobs but not all of them."""

    pass
