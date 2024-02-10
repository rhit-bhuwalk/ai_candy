import hydra
from src.models.gpt4all_model import MyGPT4ALL
from langchain.chains import LLMChain
from langchain import PromptTemplate
@hydra.main(config_path='./configs', config_name='config')
def main(cfg):
    # instantiate the model and populate the arguments using hydra

    chat_model = MyGPT4ALL(
        model_folder_path=cfg.model.gpt4all_model.gpt4all_model_folder_path,
        model_name=cfg.model.gpt4all_model.gpt4all_model_name,
        allow_download=cfg.model.gpt4all_model.gpt4all_allow_downloading,
        allow_streaming=cfg.model.gpt4all_model.gpt4all_allow_streaming,       
    )

    prompt = PromptTemplate(template="""You are a prompt engineering assistant, with a focus on optimizing prompts for generating high-quality images. Your task is to assess and refine the given user prompt. 

User's Original Prompt:
- Prompt: {user_prompt}

Evaluation Criteria:
1. Word Count: Assess whether the length of the prompt contributes to or detracts from its effectiveness for image generation.
2. Specificity: Evaluate the level of detail and specificity. A more specific prompt typically leads to more precise and engaging images.

Prompt Rating:
- Overall Rating: [1-10]. Provide a brief explanation for your rating, focusing on the prompt's word count and specificity in relation to image generation quality.

Suggestions for Improvement:
Remember, the goal is to create vivid, detailed descriptions ideal for visual interpretation. Avoid starting with "describe" or "write" to ensure the prompts are clearly aimed at generating images.

Enhanced Prompt Examples:
Generate three alternative prompts that increase the original's specificity and detail, tailored for generating vivid and engaging images. Consider elements such as setting, mood, time of day, and unique characteristics.

   - Example enhanced prompt: "A tranquil forest at sunrise, with light filtering through dense foliage, illuminating a clear stream winding through moss-covered rocks."
   - Example enhanced prompt: "A bustling medieval marketplace at midday, filled with vendors selling colorful spices, fabrics, and handcrafted goods under a bright blue sky."
   - Example enhanced prompt: "An abandoned Victorian mansion under the full moon, with ivy climbing its stone walls and a mysterious light glowing from an upper window."
""", input_variables=["user_prompt"],
)

    chain = LLMChain(llm=chat_model, prompt=prompt)
    while True:
        user_prompt = input('Enter your Query: ')
        if user_prompt == 'exit':
            break
        
        # Process the query through the chain and print the response
        response = chain.invoke({'user_prompt': user_prompt}, **{
    'n_predict': cfg.model.gpt4all_model.gpt4all_n_predict,
    'temp': cfg.model.gpt4all_model.gpt4all_temperature,
    'top_p': cfg.model.gpt4all_model.gpt4all_top_p,
    'top_k': cfg.model.gpt4all_model.gpt4all_top_k,
    'n_batch': cfg.model.gpt4all_model.gpt4all_n_batch,
    'repeat_last_n': cfg.model.gpt4all_model.gpt4all_repeat_last_n,
    'repeat_penalty': cfg.model.gpt4all_model.gpt4all_penalty,
    'max_tokens': cfg.model.gpt4all_model.gpt4all_max_tokens,
})
        print(response)


if __name__ == '__main__':
    main()