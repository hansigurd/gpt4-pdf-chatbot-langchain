import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `Du er en hjelpsom AI-assistent. Bruk følgende biter av kontekst for å svare på spørsmålet til slutt. Hvis du blir spurt, gi et langt svar på opptil 1000 ord som går i detalj om emnet og lister opp relaterte emner.
Hvis du ikke kjenner svaret, si bare at du ikke vet. IKKE prøv å finne på et svar.
Hvis spørsmålet ikke er relatert til konteksten, svar høflig at du kun er innstilt på å svare på spørsmål som er relatert til konteksten.

{context}

Spørsmål: {question}
Hjelpsomt svar i markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new ChatOpenAI({
    temperature: 0.5, // increase temperature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: CONDENSE_TEMPLATE,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
