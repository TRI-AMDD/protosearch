import sys
import shutil
import os
import numpy as np
import tempfile
import unittest
from ase.db import connect

from protosearch.workflow.workflow import Workflow
from protosearch.workflow.standard_states import StandardStates


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_workflow_submit(self):
        WF = Workflow()
        prototype = {'spacegroup': 221,
                     'wyckoffs': ['a', 'd'],
                     'species': ['Ru', 'O']}
        WF.submit(prototype)

    def test_standard_states(self, elements=['K', 'Al']):
        SS = StandardStates()
        SS.submit_standard_states(elements=elements)

    def test_job_status(self):
        WF = Workflow()
        WF.check_submissions()


if __name__ == '__main__':
    unittest.main()
