# README: IHEC Student Chatbot Project

## Table of Contents
1. [Project Context and Objectives](#project-context-and-objectives)
   - [Context](#context)
   - [Objectives](#objectives)
2. [Functional Scope](#functional-scope)
   - [Core Features](#core-features)
   - [Use Cases](#use-cases)
   - [Optional Features](#optional-features)
3. [Technical Constraints](#technical-constraints)
   - [Website Integration](#website-integration)
   - [Data Storage](#data-storage)
   - [Security Standards](#security-standards)
   - [Performance and Accessibility](#performance-and-accessibility)
4. [Design and UX/UI](#design-and-uxui)
   - [Graphic Guidelines](#graphic-guidelines)
   - [Ergonomics](#ergonomics)
5. [Deliverables](#deliverables)

---

## Project Context and Objectives

### Context
The **Institut des Hautes Études Commerciales (IHEC)** aims to integrate an intelligent chatbot into its current website to efficiently address frequently asked questions from students. These questions may relate to administration, academic programs, registration, schedules, exams, and more.

### Objectives
- **Facilitate access to information** for students.
- Provide a **fast, 24/7, and user-friendly experience**.
- Ensure the chatbot adheres to the website’s **graphic guidelines** and **security standards**.

---

## Functional Scope

### Core Features
- **Automated responses** to frequently asked questions (FAQ).
- **Guided navigation** to direct students to relevant resources (e.g., documents, forms).
- **Keyword-based search system** (e.g., "schedule," "registration").
- **Multilingual support** (French and English, based on IHEC’s needs).

### Use Cases
1. **Common Question**:  
   *"What documents are required for registration?"*  
   The chatbot should provide a precise answer and direct the user to the appropriate section of the website.

2. **Advanced Search**:  
   *"I want to view the Marketing program curriculum."*  
   The chatbot should provide a direct link to the relevant program.

### Optional Features
- Integration of a **feedback system** to improve chatbot performance.

---

## Technical Constraints

### Website Integration
- The chatbot must seamlessly integrate into the **existing IHEC website**, adhering to its graphic guidelines (fonts, colors, logos).
- No dependency on external services like OpenAI, Gemini, or other non-locally hosted APIs.
- Limited hardware resources.

### Data Storage
- Use of **CSV files** or **JSON objects** for data storage.

### Security Standards
- Compliance with **GDPR** for user data protection.
- Encryption of sensitive data.
- No storage of user data (e.g., email, phone number).
- Restricted access to collected data, limited to authorized administrators.

### Performance and Accessibility
- Fast loading on both **mobile** and **desktop** devices.
- Compatibility with major browsers (Chrome, Firefox, Safari, Edge).
- **Responsive** and intuitive interface.

---

## Design and UX/UI

### Graphic Guidelines
- Strict adherence to IHEC’s **color scheme**, **fonts**, and **graphic styles**.
- A **floating button** (or visible in a corner of the page) to access the chatbot.

### Ergonomics
- Simple and intuitive interface.
- Responses displayed in an **interactive dialogue format**.
- Option to go back or rephrase a question.

---

## Deliverables
- A fully **operational chatbot** that can be easily integrated into the IHEC website.
- **Technical documentation** for maintenance and future updates.
- A **user manual** for IHEC administrative staff.