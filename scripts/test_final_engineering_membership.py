import unittest
from scripts.freeze_final_engineering_evaluation import choose_members


class FinalMembershipTests(unittest.TestCase):
    def test_patient_grouping_precedes_variants_and_does_not_depend_on_record_order(self):
        records=['patient001/z','patient001/a','patient002/a','patient003/a','patient004/a']
        a=choose_members(records,{'patient002'},3,'fixed')
        self.assertEqual(a,choose_members(list(reversed(records)),{'patient002'},3,'fixed'))
        self.assertEqual(len({m['patientId'] for m in a}),3)
        self.assertNotIn('patient002',{m['patientId'] for m in a})
        self.assertEqual(next(m['record'] for m in a if m['patientId']=='patient001'),'patient001/a')
        with self.assertRaisesRegex(ValueError,'Insufficient'):choose_members(records,{'patient002'},4,'fixed')


if __name__=='__main__':unittest.main()
