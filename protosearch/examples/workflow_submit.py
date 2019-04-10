from protosearch.workflow.workflow import Workflow

WF = Workflow()

prototype = {'spacegroup':221,
             'wyckoffs':['a', 'd'],
             'species':['Ru', 'O']}

WF.submit(prototype)
