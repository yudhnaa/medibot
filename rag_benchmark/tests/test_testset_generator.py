from django.test import SimpleTestCase

from rag_benchmark.services.testset_generator import RagasBenchmarkDatasetGenerator


class RagasBenchmarkDatasetGeneratorTests(SimpleTestCase):
    def setUp(self):
        self.generator = RagasBenchmarkDatasetGenerator()

    def test_infer_scenario_prioritizes_negation(self):
        scenario = self.generator._infer_scenario(
            question="Tôi ho nhưng không sốt, có phải covid không?",
            query_style="WEB_SEARCH_LIKE",
            query_length="MEDIUM",
        )
        self.assertEqual(scenario, "negation")

    def test_infer_scenario_maps_noisy_query_style(self):
        scenario = self.generator._infer_scenario(
            question="em bi sot vs ho @@",
            query_style="POOR_GRAMMAR",
            query_length="SHORT",
        )
        self.assertEqual(scenario, "noisy_query")

    def test_build_case_payload_maps_required_schema(self):
        row = {
            "user_input": "Cancer a risk for covid?",
            "reference_contexts": [
                "Title: coronavirus disease (covid-19)\n"
                "Section[risk]: older people cardiovascular disease diabetes "
                "chronic respiratory disease cancer\n"
                "Section[symptom]: fever cough tiredness"
            ],
            "reference": "Yes, cancer is listed as a risk factor.",
            "query_style": "POOR_GRAMMAR",
            "query_length": "SHORT",
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        }

        case = self.generator._build_case_payload(
            row=row,
            case_no=7,
            split="dev",
            dataset_version="v1",
            fallback_title="fallback-title",
        )

        self.assertEqual(case["case_id"], "dev-noisy_query-007")
        self.assertEqual(case["dataset_version"], "v1")
        self.assertEqual(case["split"], "dev")
        self.assertEqual(case["gold_titles"], ["coronavirus disease (covid-19)"])
        self.assertEqual(case["expected_mode"], "single-disease")
        self.assertEqual(case["expected_behavior"], "answer")
        self.assertEqual(case["must_have_sections"], ["risk", "symptom"])

    def test_resolve_provider_accepts_openrouter(self):
        provider = self.generator._resolve_provider("openrouter")
        self.assertEqual(provider, "openrouter")

    def test_resolve_provider_rejects_unknown_provider(self):
        with self.assertRaises(ValueError):
            self.generator._resolve_provider("azure-openai")
