import http from 'k6/http';
import { sleep } from 'k6';

const queriesData = [
  "Explain the theory of relativity in simple terms.",
  "What are the key differences between machine learning and deep learning?",
  "Write a short story about a dragon and a knight.",
  "Translate 'Hello, how are you?' into French.",
  "Summarize the book '1984' by George Orwell.",
  "What is the capital of France?",
  "List 5 ways to improve time management skills.",
  "Describe the process of photosynthesis.",
  "Give me a recipe for chocolate chip cookies.",
  "What's the current weather in New York City?",
  "Who won the FIFA World Cup in 2022?",
  "What is the purpose of life?",
  "Generate a random joke for me.",
  "Explain quantum mechanics to a 5-year-old.",
  "What is artificial intelligence?",
  "How does a blockchain work?",
  "Describe the Great Wall of China.",
  "Give me a list of the top 10 movies of all time.",
  "How can I learn Python programming quickly?",
  "What are the latest advancements in space exploration?",
  "Tell me a motivational quote.",
  "Write a haiku about the ocean.",
  "How can I improve my public speaking skills?",
  "What are the symptoms of COVID-19?",
  "Explain how neural networks work.",
  "What is the history of the Internet?",
  "Write a poem about autumn.",
  "What's the best way to learn a new language?",
  "What is the difference between a star and a planet?",
  "Explain the concept of supply and demand.",
  "Describe the human immune system.",
  "What are the benefits of meditation?",
  "Who was Albert Einstein?",
  "What's the tallest mountain in the world?",
  "How do airplanes fly?",
  "What is the meaning of the word 'serendipity'?",
  "How can I be more productive during the day?",
  "What is climate change and how does it affect the planet?",
  "Write a dialogue between two robots discussing the future of technology.",
  "What is the significance of the Mona Lisa painting?",
  "Explain how electricity works in a simple way.",
  "What are the benefits of renewable energy?",
  "Tell me a fun fact about space.",
  "What is the Turing test?",
  "How do vaccines work?",
  "Write a brief history of ancient Rome.",
  "What is the Fibonacci sequence?",
  "How does the brain process emotions?",
  "What are the stages of the water cycle?",
  "Explain string theory in one sentence.",
  "Generate a Flask App",
  "Generate a deployment file for kubernetes",
  "Generate an express Js app",
  "Generate table of students with fake particulars"
]

const longChatHistory = `
<|start_header_id|>user<|end_header_id|>
Hello, how are you?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
I'm good, how can I help you?
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Can you explain the process of photosynthesis in detail? I’m curious about how plants convert light into energy and what factors can influence this process. 
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. It occurs primarily in the chloroplasts of plant cells and involves two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). During the light-dependent reactions, chlorophyll captures sunlight, and water molecules are split to release oxygen while generating ATP and NADPH. In the Calvin cycle, carbon dioxide is fixed and converted into glucose using the energy from ATP and NADPH. This process is crucial for life on Earth, providing the oxygen we breathe and serving as the foundation of the food chain.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What are the main factors affecting photosynthesis? Could you also give some examples?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
The main factors affecting photosynthesis include light intensity, carbon dioxide concentration, temperature, and water availability. Higher light intensity generally increases the rate of photosynthesis up to a certain point. Similarly, increased levels of carbon dioxide can enhance photosynthesis, but only if other conditions, such as light and temperature, are optimal. Temperature influences the enzyme activity involved in photosynthesis, while water is essential for the light-dependent reactions. For example, during a drought, water scarcity can limit photosynthesis, leading to reduced plant growth.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Can you explain how chlorophyll works in this process?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
`

const mediumChatHistory = `
<|start_header_id|>user<|end_header_id|>
Hello, how are you?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
I'm good, how can I help you?
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What are some common programming languages?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Some common programming languages include Python, Java, JavaScript, C++, and Ruby. Python is known for its readability and is widely used in data science and web development. Java is popular for building enterprise-level applications. JavaScript is essential for web development, enabling dynamic content on websites.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Which language is best for beginners?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
`


const shortChatHistory = `
<|start_header_id|>user<|end_header_id|>
Hello, how are you?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
I'm good, how can I help you?
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What is the capital of Japan?
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
`

const chatHistoryList = [longChatHistory, mediumChatHistory, shortChatHistory]


function generateTemplate(chat_history, user_question) {
  return `
<|begin_of_text|>
<|start_header_id|>assistant<|end_header_id|>
You are a helpful assistant. Answer the user’s latest question clearly and succinctly with sufficent detail.
After responding to the first question, wait for more User input.
<|eot_id|>

${chat_history}

<|start_header_id|>user<|end_header_id|>
${user_question}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
  `;
}

// Seeded random number generator
function seededRandom(seed) {
  // Simple PRNG based on the seed
  const m = 0x80000000; // 2**31
  const a = 1103515245; // Multiplicative factor
  const c = 12345; // Increment
  seed = (a * seed + c) % m;
  return seed / m; // Return a float between 0 and 1
}

function getRandomQuery(queries, seed) {
  const randomValue = seededRandom(seed);
  const randomIndex = Math.floor(randomValue * queries.length);
  return queries[randomIndex];
}

// These are the commandline options,
// These are the commandline options,
export const options = {
  stages: [
    { duration: '5m', target: 10 },
    { duration: '20m', target: 10 },
    { duration: '5m', target: 0 },
  ],
};

// Called once per VU
export default function() {

  // Define the API provider and model URL
  const VLLM_URL = "http://triton-route-triton-inference-services.apps.nebula.sl"
  const VLLM_FULL_MODEL = "/v2/models/meta-llama-3-8b-instruct-awq/generate"

  const userQuery = getRandomQuery(queriesData, 2)
  const chatHistory = getRandomQuery(chatHistoryList, 2)
  

  const query = generateTemplate(chatHistory, userQuery)

  const payload = {
    "text_input": query,
    "parameters": {"stream": false, "temperature": 0, "max_tokens": 1000},
    "exclude_input_in_output": true
  }

  const params = {
    headers: {
      'Content-Type': 'application/json'
    },
  }
    
  const response = http.post(`${VLLM_URL}${VLLM_FULL_MODEL}`, JSON.stringify(payload), params)

    // Wait for a random time between 5 to 15 seconds before the next request (llm simulation)
    sleep(randomIntBetween(5, 15));
}