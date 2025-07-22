test_config_dict = {'mistral_large': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_mistral',
                                      'exp_name': 'custom_set_xling_mistral',
                                      'langchain_project': 'mistral_large_custom_set_de',
                                      'final_call_model': 'Mistral-large',
                                      },

                    'ada_4o_xling_wt': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_ada_4o_wt',
                                                          'exp_name': 'custom_set_xling_ada_4o_wt',
                                                          'langchain_project': 'ada_4o_custom_set_de_wt',
                                                          'final_call_model': 'gpt-4o',
                                                          'with_translation': "true"
                                                          },
                    'ada_4o_xling_wot': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_ada_4o_wot',
                                        'exp_name': 'custom_set_xling_ada_4o_wot',
                                        'langchain_project': 'ada_4o_custom_set_de_wot',
                                        'final_call_model': 'gpt-4o',
                                        'with_translation': "false"
                                        },
                    'embed_small_4o_xling_wt': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_small_4o_wt',
                                        'exp_name': 'custom_set_xling_emebed_small_4o_wt',
                                        'langchain_project': 'embedding_small_4o_custom_set_de_wt',
                                        'final_call_model': 'gpt-4o',
                                        'with_translation': "true"
                                        },

                    'embed_small_4o_xling_wot': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_small_4o_wot',
                                        'exp_name': 'custom_set_xling_emebed_small_4o_wot',
                                        'langchain_project': 'embedding_small_4o_custom_set_de_wot',
                                        'final_call_model': 'gpt-4o',
                                        'with_translation': "false"
                                        },

                    'embed_large_4o_xling_wt': {'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_large_4o_wt',
                                        'exp_name': 'custom_set_xling_emebed_large_4o_wt',
                                        'langchain_project': 'embedding_large_4o_custom_set_de_wt',
                                        'final_call_model': 'gpt-4o',
                                        'with_translation': "true"
                                        },
                    'embed_large_4o_xling_wot': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_large_4o_wot',
                        'exp_name': 'custom_set_xling_emebed_large_4o_wot',
                        'langchain_project': 'embedding_large_4o_custom_set_de_wot',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "false"
                        },
                    'embed_large_4o_xling_wt_embed_large_1536': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_large_1536_4o_wt',
                        'exp_name': 'custom_set_xling_emebed_large_1536_4o_wt',
                        'langchain_project': 'embedding_large_1536_4o_custom_set_de_wt',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "true"
                    },
                    'embed_large_4o_xling_wot_embed_large_1536': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_de_emebed_large_1536_4o_wot',
                        'exp_name': 'custom_set_xling_emebed_large_1536_4o_wot',
                        'langchain_project': 'embedding_large_1536_4o_custom_set_de_wot',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "false"
                    },
                    'ada_4o_xling_wot_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_ada_4o_xling_wot_xset',
                        'exp_name': 'custom_set_ada_4o_xling_wot_xset',
                        'langchain_project': 'ada_4o_xling_wot_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "false"
                    },
                    'ada_4o_xling_wt_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_ada_4o_xling_wt_xset',
                        'exp_name': 'custom_set_ada_4o_xling_wt_xset',
                        'langchain_project': 'ada_4o_xling_wt_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "true"
                    },

                    'embedding_small_4o_xling_wot_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_embedding_small_4o_xling_wot_xset',
                        'exp_name': 'custom_set_embedding_small_4o_xling_wot_xset',
                        'langchain_project': 'embedding_small_4o_xling_wot_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "false"
                    },
                    'embedding_small_4o_xling_wt_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_embedding_small_4o_xling_wt_xset',
                        'exp_name': 'custom_set_embedding_small_4o_xling_wt_xset',
                        'langchain_project': 'embedding_small_4o_xling_wt_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "true"
                    },

                    'embedding_large_4o_xling_wot_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_embedding_large_4o_xling_wot_xset',
                        'exp_name': 'custom_set_embedding_large_4o_xling_wot_xset',
                        'langchain_project': 'embedding_large_4o_xling_wot_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "false"
                    },
                    'embedding_large_4o_xling_wt_xset': {
                        'res_file_name': 'results/cross_lingual/qa_responses_custom_set_embedding_large_4o_xling_wt_xset',
                        'exp_name': 'custom_set_embedding_large_4o_xling_wt_xset',
                        'langchain_project': 'embedding_large_4o_xling_wt_xset',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "true"
                    },
                    'test_config': {
                        'res_file_name': 'results/cross_lingual/test_config',
                        'exp_name': 'test_config',
                        'langchain_project': 'test_config',
                        'final_call_model': 'gpt-4o',
                        'with_translation': "true"
                    },

                    }
