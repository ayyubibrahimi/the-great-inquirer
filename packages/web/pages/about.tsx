"use client";
import React from "react";
import RootLayout from "../app/layout";

const AboutPage: React.FC = () => {
  return (
    <RootLayout>
      <section className="space-y-4 p-6">
        <h2 className="text-2xl font-bold">About The Great Inquirer</h2>
        <p>
          The Great Inquirer is a groundbreaking platform designed to foster
          direct communication between the residents of New Orleans and their
          city council representatives. Our mission is to empower the community
          by providing a seamless and efficient channel for citizens to express
          their concerns, ask questions, and receive accurate information
          directly from their elected officials.
        </p>
        <h2 className="text-2xl font-bold">Features</h2>
        <ul className="list-disc pl-5">
          <li>
            Direct Communication: The Great Inquirer facilitates direct and
            transparent communication between New Orleanians and city council
            representatives. By eliminating barriers and intermediaries, we
            enable residents to engage in meaningful conversations with their
            representatives, fostering a sense of collaboration and
            accountability.
          </li>
          <li>
            AI-Powered Assistance: Our platform leverages advanced AI
            technologies to enhance the user experience. We employ cutting-edge
            language models to provide intelligent responses and assist in
            finding relevant information from a vast collection of New Orleans
            City Council meeting transcripts.
          </li>
          <li>
            Contextual Dialogue Recreation: Through our AI assistant, The Great
            Inquirer recreates the actual dialogue that took place during city
            council meetings. By analyzing transcripts and generating
            comprehensive statements and responses, we ensure that users receive
            a holistic understanding of the discussions and decisions made by
            council members and law enforcement stakeholders.
          </li>
          <li>
            Comprehensive Answers: The Great Inquirer aims to provide accurate
            and comprehensive answers to user queries. Our AI assistant
            generates summaries of statements and responses from city council
            members and law enforcement stakeholders, accompanied by direct
            quotes from the meeting transcripts. We strive to preserve context
            and substance while addressing the questions posed by our users.
          </li>
          <li>
            Reliable Information: The Great Inquirer is built upon a vast
            database of New Orleans criminal justice city council meetings.
            These meetings were meticulously collected from YouTube, ensuring
            that the information provided is reliable and authentic.
          </li>
        </ul>
        <p>
          We are continuously working to improve and expand The Great
          Inquirer&apos;s capabilities, with a focus on enhancing user
          experience, incorporating user feedback, and expanding the coverage of
          available information. Our vision is to foster a vibrant and engaged
          community that actively participates in the decision-making processes
          that shape the city of New Orleans.
        </p>
        <p>
          Join The Great Inquirer today and be part of the conversation that
          shapes your community&apos;s future. Together, we can make a
          difference.
        </p>
        <p>
          Note: The Great Inquirer is currently a work in progress, and we
          appreciate your patience and support as we strive to deliver the best
          possible experience for our users.
        </p>
      </section>
    </RootLayout>
  );
};

export default AboutPage;