## Job Scraping: Finding the Most Important Skills for a Given Role

<ol>
    <li>Pre-Ingest a large number of data job postings from LinkedIn/Indeed with Salaries posted</li>
    <li>Use Spacy to extract top skills from job description
    <li>Use ML to create a model for predicting salary and skills needed
    <li>User inputs a job title and location
    <li>Returns:
        <ul>
            <li>Model run results given those results
            <li>Predicted salary/salary range
            <li>Common/predicted skills in that location for that role
        </ul>
    <li>Allow raw querying through the pre-loaded database
</ol>