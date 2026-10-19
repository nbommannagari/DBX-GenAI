# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Demo - Creating a Genie Agent for Unstructured Data
# MAGIC #### Duration: 30 minutes
# MAGIC ## Overview
# MAGIC In this demo, we'll build a **Genie Agent** that answers questions over **unstructured documents** rather than SQL tables. Instead of querying rows and columns, we'll point Genie at a Unity Catalog **volume** of Airbnb PDFs, enable **content search** so Genie can index the files, and then ask natural-language questions in **Agent mode** that are answered with **citations** back to the source documents.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC By the end of this demo, you will be able to:
# MAGIC 1. Inspect the PDF documents in a Unity Catalog volume created during classroom setup
# MAGIC 2. Create a Genie Agent and attach a volume of documents as a source
# MAGIC 3. Enable content search so Genie can index and retrieve from the files
# MAGIC 4. Query the documents in Agent mode and validate cited responses
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Beta Features</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Querying files in volumes with a Genie Agent (and PDF content search) is a Databricks Beta feature. While this notebook has been thoroughly tested, full functionality is not guaranteed and UI labels may change.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Cell 3
# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 6</strong>: How to select an environment version (<a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976d2; text-decoration: underline;">GCP</a>)</li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V6</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Classroom Setup

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup-02

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Inspect the Volume in Catalog Explorer
# MAGIC
# MAGIC As part of the classroom setup, certain assets have been managed for you. 
# MAGIC
# MAGIC Two **Unity Catalog** volumes have been created with some simple PDFs:
# MAGIC
# MAGIC - `listings_vol`
# MAGIC     - `listings/airbnb_listings.pdf` &mdash; San Francisco Airbnb property listings
# MAGIC - `house_rules_vol`
# MAGIC     - `additional_listing_info/airbnb_house_rules.pdf` &mdash; guest house rules
# MAGIC
# MAGIC Navigate to the **Catalog Explorer** and search for the PDFs in your **labuser** catalog. 
# MAGIC Before wiring any documents into Genie, take a moment to explore them in the UI. 
# MAGIC
# MAGIC 1. In the left sidebar, click **Catalog**.
# MAGIC
# MAGIC 2. Navigate to your **labuser_UNIQUE_ID** catalog, **rag_schema** schema, then expand **Volumes**. 
# MAGIC
# MAGIC 3. You will see both volumes, each with their own subfolder, and each containing 1 pdf.
# MAGIC
# MAGIC 4. Click a PDF to preview it, and note the kind of content each holds (property listings vs. guest policies). You'll reference these differences when you write the source description for Genie.

# COMMAND ----------

# DBTITLE 1,Cell 7
# MAGIC %md-sandbox
# MAGIC ### B1. Prepare Metadata for Your Genie Agent
# MAGIC
# MAGIC A clear description helps Genie decide when to search your documents and how to interpret them. 
# MAGIC
# MAGIC Read the two descriptions below &mdash; one per document set. 
# MAGIC
# MAGIC Each block has a **Copy** button.  You'll paste the relevant text into the **Description** field when you attach the volume to your Genie Agent. 
# MAGIC
# MAGIC 1. Copy and paste the following into the `listings_vol` volume. 
# MAGIC
# MAGIC <div style="display: flex; align-items: stretch; gap: 0; margin: 16px 0; border: 1px solid #B4D5ED; border-radius: 12px; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
# MAGIC   <div style="background: #2272B4; display: flex; align-items: center; justify-content: center; padding: 16px 20px; min-width: 48px;">
# MAGIC     <svg viewBox="0 0 24 24" style="width: 44px; height: 44px; fill: none; stroke: #fff; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;">
# MAGIC       <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
# MAGIC       <text x="12" y="16" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif" font-size="9" font-weight="700" fill="#fff" stroke="none">1</text>
# MAGIC     </svg>
# MAGIC   </div>
# MAGIC   <div style="background: #F0F6FB; padding: 14px 18px; display: flex; flex-direction: column; justify-content: center; flex: 1; position: relative;">
# MAGIC     <button onclick="(function(b){var p=b.parentElement,c=p.querySelector('.bh-copy-text');if(!c)return;var v=c.textContent.trim(),t=document.createElement('textarea');t.value=v;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.cp-l'),i=b.querySelector('.cp-i'),k=b.querySelector('.cp-k');i.style.display='none';k.style.display='';l.textContent='Copied!';b.style.color='#2272B4';setTimeout(function(){i.style.display='';k.style.display='none';l.textContent='Copy';b.style.color='#5A6F77'},2000)}})(this)" style="position:absolute;top:14px;right:14px;display:inline-flex;align-items:center;gap:4px;background:none;border:1px solid #DCE0E2;color:#5A6F77;cursor:pointer;padding:5px 10px;border-radius:6px;font-size:13px;font-weight:600;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;transition:color 0.15s,background 0.15s" onmouseover="this.style.background='rgba(0,0,0,0.04)'" onmouseout="this.style.background='none'"><svg class="cp-i" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:16px;height:16px" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg><svg class="cp-k" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:16px;height:16px;display:none" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span class="cp-l">Copy</span></button>
# MAGIC     <span style="font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #1B5162; margin-bottom: 4px;">Airbnb Listings</span>
# MAGIC     <p style="margin: 0 0 10px 0; color: #1B3139; font-size: 14px; line-height: 1.5;">
# MAGIC       Reference catalog of San Francisco Airbnb property listings. One PDF stored in the following folder:
# MAGIC     </p>
# MAGIC     <ul style="margin: 0 0 12px 0; padding-left: 20px; color: #1B3139; font-size: 14px; line-height: 1.5; list-style: none;">
# MAGIC       <li style="margin-bottom: 6px;"><code style="background: #dce8f0; padding: 1px 5px; border-radius: 4px; font-size: 13px;">listings/</code> &mdash; Individual property listings including property name, neighborhood, property type, nightly price, key amenities, proximity to transit and landmarks, and host information.</li>
# MAGIC     </ul>
# MAGIC     <p style="margin: 0 0 6px 0; color: #1B3139; font-size: 14px; line-height: 1.5; font-weight: 600;">Use this document to answer questions about:</p>
# MAGIC     <p style="margin: 0 0 10px 0; color: #1B3139; font-size: 14px; line-height: 1.5;">
# MAGIC       Property descriptions &middot; Neighborhoods &middot; Amenities &middot; Nightly pricing &middot; Transit and landmark proximity &middot; Hosts
# MAGIC     </p>
# MAGIC     <p style="margin: 0; color: #5A6F77; font-size: 13px; line-height: 1.5; font-style: italic;">
# MAGIC       This is a listings catalog. It does not contain guest policies or house rules.
# MAGIC     </p>
# MAGIC     <div class="bh-copy-text" style="display:none">Airbnb Listings
# MAGIC
# MAGIC Reference catalog of San Francisco Airbnb property listings. One PDF stored in the following folder:
# MAGIC
# MAGIC listings/ &mdash; Individual property listings including property name, neighborhood, property type, nightly price, key amenities, proximity to transit and landmarks, and host information.
# MAGIC
# MAGIC Use this document to answer questions about:
# MAGIC Property descriptions &middot; Neighborhoods &middot; Amenities &middot; Nightly pricing &middot; Transit and landmark proximity &middot; Hosts
# MAGIC
# MAGIC This is a listings catalog. It does not contain guest policies or house rules.</div>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC 2. Copy and paste the following into the `house_rules_vol` volume. 
# MAGIC
# MAGIC <div style="display: flex; align-items: stretch; gap: 0; margin: 16px 0; border: 1px solid #B4D5ED; border-radius: 12px; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
# MAGIC   <div style="background: #2272B4; display: flex; align-items: center; justify-content: center; padding: 16px 20px; min-width: 48px;">
# MAGIC     <svg viewBox="0 0 24 24" style="width: 44px; height: 44px; fill: none; stroke: #fff; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;">
# MAGIC       <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>
# MAGIC       <text x="12" y="16" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif" font-size="9" font-weight="700" fill="#fff" stroke="none">2</text>
# MAGIC     </svg>
# MAGIC   </div>
# MAGIC   <div style="background: #F0F6FB; padding: 14px 18px; display: flex; flex-direction: column; justify-content: center; flex: 1; position: relative;">
# MAGIC     <button onclick="(function(b){var p=b.parentElement,c=p.querySelector('.bh-copy-text');if(!c)return;var v=c.textContent.trim(),t=document.createElement('textarea');t.value=v;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.cp-l'),i=b.querySelector('.cp-i'),k=b.querySelector('.cp-k');i.style.display='none';k.style.display='';l.textContent='Copied!';b.style.color='#2272B4';setTimeout(function(){i.style.display='';k.style.display='none';l.textContent='Copy';b.style.color='#5A6F77'},2000)}})(this)" style="position:absolute;top:14px;right:14px;display:inline-flex;align-items:center;gap:4px;background:none;border:1px solid #DCE0E2;color:#5A6F77;cursor:pointer;padding:5px 10px;border-radius:6px;font-size:13px;font-weight:600;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;transition:color 0.15s,background 0.15s" onmouseover="this.style.background='rgba(0,0,0,0.04)'" onmouseout="this.style.background='none'"><svg class="cp-i" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:16px;height:16px" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg><svg class="cp-k" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="width:16px;height:16px;display:none" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span class="cp-l">Copy</span></button>
# MAGIC     <span style="font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #1B5162; margin-bottom: 4px;">Airbnb House Rules</span>
# MAGIC     <p style="margin: 0 0 10px 0; color: #1B3139; font-size: 14px; line-height: 1.5;">
# MAGIC       Guest-facing house rules for Airbnb stays. One PDF stored in the following folder:
# MAGIC     </p>
# MAGIC     <ul style="margin: 0 0 12px 0; padding-left: 20px; color: #1B3139; font-size: 14px; line-height: 1.5; list-style: none;">
# MAGIC       <li style="margin-bottom: 6px;"><code style="background: #dce8f0; padding: 1px 5px; border-radius: 4px; font-size: 13px;">additional_listing_info/</code> &mdash; House rules covering check-in and check-out times, quiet hours, pet policy, smoking policy, maximum guest counts, parties and events, and damage and liability expectations.</li>
# MAGIC     </ul>
# MAGIC     <p style="margin: 0 0 6px 0; color: #1B3139; font-size: 14px; line-height: 1.5; font-weight: 600;">Use this document to answer questions about:</p>
# MAGIC     <p style="margin: 0 0 10px 0; color: #1B3139; font-size: 14px; line-height: 1.5;">
# MAGIC       Check-in / check-out &middot; Quiet hours &middot; Pet and smoking policies &middot; Guest limits &middot; Parties and events &middot; Guest conduct
# MAGIC     </p>
# MAGIC     <p style="margin: 0; color: #5A6F77; font-size: 13px; line-height: 1.5; font-style: italic;">
# MAGIC       This is a policy document. It does not contain property listings or pricing.
# MAGIC     </p>
# MAGIC     <div class="bh-copy-text" style="display:none">Airbnb House Rules
# MAGIC
# MAGIC Guest-facing house rules for Airbnb stays. One PDF stored in the following folder:
# MAGIC
# MAGIC additional_listing_info/ &mdash; House rules covering check-in and check-out times, quiet hours, pet policy, smoking policy, maximum guest counts, parties and events, and damage and liability expectations.
# MAGIC
# MAGIC Use this document to answer questions about:
# MAGIC Check-in / check-out &middot; Quiet hours &middot; Pet and smoking policies &middot; Guest limits &middot; Parties and events &middot; Guest conduct
# MAGIC
# MAGIC This is a policy document. It does not contain property listings or pricing.</div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Cell 10
# MAGIC %md-sandbox
# MAGIC ## C. Create the Genie Agent
# MAGIC
# MAGIC Now create the Genie Agent that will answer questions about these documents stored in one of your volumes.
# MAGIC
# MAGIC 1. In the left sidebar, click **Genie Agents**
# MAGIC
# MAGIC 2. Click **+ New** to start a new Genie Agent
# MAGIC
# MAGIC 3. Select `labuser_UNIQUE_ID.rag_schema.house_rules_vol`
# MAGIC
# MAGIC 4. Select **Create**
# MAGIC
# MAGIC 5. Click **Configure <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style="vertical-align: -3px; display: inline-block;"><path fill-rule="evenodd" clip-rule="evenodd" d="M7.01548 3.32805L7.61271 1.50939C7.72946 1.50316 7.84708 1.5 7.9655 1.5C8.08391 1.5 8.20153 1.50316 8.31826 1.50939L8.91548 3.32804C8.9891 3.55221 9.16392 3.72868 9.3874 3.80438C9.73161 3.92099 10.0571 4.07892 10.3578 4.27242C10.5564 4.40019 10.8036 4.42696 11.025 4.34468L12.82 3.6774C12.9764 3.8529 13.1234 4.03691 13.2602 4.22863L12.2105 5.82982C12.0811 6.02713 12.0522 6.27382 12.1323 6.49574C12.2525 6.82878 12.334 7.18058 12.3712 7.54597C12.395 7.78097 12.5282 7.99099 12.7306 8.11277L14.3728 9.10092C14.333 9.33452 14.2807 9.56379 14.2167 9.78806L12.308 9.96587C12.0734 9.98773 11.8627 10.1186 11.7391 10.3192C11.5496 10.6268 11.3234 10.9096 11.0666 11.1616C10.8982 11.3267 10.8173 11.5615 10.8483 11.7953L11.0997 13.696C10.8946 13.8091 10.6828 13.9114 10.4651 14.0022L9.13584 12.6206C8.97252 12.4509 8.73894 12.3677 8.50509 12.3961C8.32852 12.4175 8.14847 12.4285 7.96551 12.4285C7.78253 12.4285 7.60247 12.4175 7.42589 12.3961C7.19204 12.3677 6.95846 12.4509 6.79514 12.6206L5.46589 14.0022C5.24816 13.9114 5.03638 13.8091 4.83131 13.696L5.08272 11.7953C5.11365 11.5615 5.03278 11.3267 4.86442 11.1615C4.60761 10.9096 4.38144 10.6268 4.19195 10.3192C4.06833 10.1186 3.85762 9.98773 3.62299 9.96587L1.71433 9.78806C1.65032 9.56379 1.59804 9.33451 1.55822 9.1009L3.2004 8.11275C3.40279 7.99096 3.53599 7.78095 3.55987 7.54595C3.597 7.18058 3.67847 6.8288 3.79872 6.49578C3.87884 6.27386 3.84988 6.02718 3.72051 5.82986L2.67076 4.22865C2.80759 4.03693 2.9546 3.85292 3.111 3.67741L4.90605 4.34469C5.1274 4.42697 5.37457 4.4002 5.57317 4.27243C5.87394 4.07893 6.19937 3.92101 6.54357 3.8044C6.76705 3.72869 6.94187 3.55223 7.01548 3.32805ZM7.9655 0C7.62552 0 7.29014 0.0212511 6.9607 0.0625736C6.67304 0.0986558 6.43194 0.2973 6.34148 0.572745L5.7024 2.51888C5.49244 2.60565 5.2886 2.70411 5.0917 2.81341L3.17198 2.09979C2.90054 1.99888 2.59526 2.06326 2.38769 2.26519C1.9069 2.73289 1.48464 3.26068 1.13301 3.83661C0.981982 4.08399 0.987006 4.39625 1.14592 4.63864L2.26919 6.35198C2.20692 6.56757 2.15656 6.78814 2.11885 7.01293L0.363322 8.06928C0.115483 8.21842 -0.0245086 8.49678 0.00354929 8.78466C0.0700536 9.46702 0.222365 10.1248 0.449669 10.7466C0.549085 11.0186 0.796189 11.209 1.08451 11.2359L3.12667 11.4261C3.25735 11.6104 3.39826 11.7868 3.54858 11.9546L3.27974 13.9872C3.2418 14.274 3.37222 14.5571 3.6149 14.7146C4.17427 15.0777 4.78216 15.373 5.42686 15.5886C5.70147 15.6804 6.00442 15.606 6.20518 15.3973L7.62743 13.919C7.73939 13.9254 7.85211 13.9285 7.96551 13.9285C8.07889 13.9285 8.1916 13.9254 8.30354 13.919L9.7258 15.3973C9.92656 15.606 10.2295 15.6804 10.5041 15.5886C11.1488 15.373 11.7567 15.0777 12.3161 14.7146C12.5588 14.5571 12.6892 14.274 12.6512 13.9872L12.3824 11.9547C12.5327 11.7869 12.6737 11.6104 12.8043 11.4261L14.8465 11.2359C15.1348 11.209 15.3819 11.0186 15.4813 10.7466C15.7086 10.1248 15.8609 9.46704 15.9275 8.78469C15.9555 8.4968 15.8155 8.21844 15.5677 8.0693L13.8122 7.01296C13.7745 6.78815 13.7241 6.56754 13.6618 6.35193L14.7851 4.63861C14.944 4.39622 14.949 4.08396 14.798 3.83658C14.4463 3.26065 14.0241 2.73287 13.5433 2.26518C13.3357 2.06325 13.0305 1.99887 12.759 2.09978L10.8393 2.8134C10.6424 2.70409 10.4385 2.60563 10.2286 2.51885L9.58949 0.572741C9.49903 0.297294 9.25793 0.0986496 8.97026 0.0625686C8.64083 0.0212494 8.30548 0 7.9655 0ZM6.48433 8C6.48433 7.17157 7.1559 6.5 7.98433 6.5C8.81275 6.5 9.48433 7.17157 9.48433 8C9.48433 8.82843 8.81275 9.5 7.98433 9.5C7.1559 9.5 6.48433 8.82843 6.48433 8ZM7.98433 5C6.32747 5 4.98433 6.34315 4.98433 8C4.98433 9.65685 6.32747 11 7.98433 11C9.64118 11 10.9843 9.65685 10.9843 8C10.9843 6.34315 9.64118 5 7.98433 5Z" fill="#0b2026"/></svg>** at the top right and scroll down to the **About this agent** section. 
# MAGIC
# MAGIC     - Click the edit icon <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style="vertical-align: -3px; display: inline-block;"><path fill-rule="evenodd" clip-rule="evenodd" d="M12.0732 2.57322C12.1709 2.47559 12.3291 2.47559 12.4268 2.57322L13.5126 3.65901C13.6102 3.75664 13.6102 3.91493 13.5126 4.01256L12 5.52513L10.5607 4.08579L12.0732 2.57322ZM9.5 5.14645L2.5 12.1464V13.5858H3.93934L10.9393 6.58579L9.5 5.14645ZM13.4874 1.51256C12.804 0.829146 11.696 0.829145 11.0126 1.51256L1.21967 11.3055C1.07902 11.4461 1 11.6369 1 11.8358V14.3358C1 14.75 1.33579 15.0858 1.75 15.0858H4.25C4.44891 15.0858 4.63968 15.0068 4.78033 14.8661L14.5732 5.07322C15.2566 4.38981 15.2566 3.28177 14.5732 2.59835L13.4874 1.51256Z" fill="#0b2026"/></svg> and rename the agent <code>⟨catalog_name⟩-genie-agent</code>. You can find your catalog name in the output from running the Classroom setup script above. 
# MAGIC     - This also has the SQL Warehouse, owner information, agent ID, Tag configuration, and the option to include a custom thumbnail.
# MAGIC     - This volume will automatically be indexed. You will see an **Indexing** message appear next to the volume. 
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <strong style="color: #0d47a1;">Requirements &amp; limits</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">For a full list of requirements and limitations when attaching volumes to a Genie Agent, see the documentation (<a href="https://docs.databricks.com/aws/en/genie-agents/volumes#limitations-for-volumes-without-content-search" style="color: #1976d2;">AWS</a> | <a href="https://learn.microsoft.com/en-us/azure/databricks/genie-agents/volumes" style="color: #1976d2;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/genie-agents/volumes#limitations-for-volumes-without-content-search" style="color: #1976d2;">GCP</a>).</p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Cell 13
# MAGIC %md-sandbox
# MAGIC ### C1. Initial Test of Genie Agent
# MAGIC
# MAGIC
# MAGIC <div style="display: flex; align-items: flex-start; gap: 24px; flex-wrap: wrap;">
# MAGIC   <div style="flex: 1; min-width: 280px;">
# MAGIC     <p>Now that your Genie Agent is created, let's verify it can see your volume. Open the agent in Agent mode and ask what files it has access to. The response should only list the single PDF in the volume we attached, which is <code>airbnb_house_rules.pdf</code>. In the chat interface, make sure <strong>Agent mode</strong> <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style="vertical-align: -3px; display: inline-block;"><path d="M6.25 8C6.25 9.41469 6.36435 10.7163 6.55762 11.6826C6.65301 12.1596 6.77495 12.5943 6.93262 12.9277C6.95991 12.9854 6.99086 13.0449 7.02539 13.1045C7.21398 13.6003 7.41403 13.9676 7.6084 14.2061C7.83569 14.4849 7.97222 14.5 8 14.5C8.02777 14.5 8.16431 14.4849 8.3916 14.2061C8.61414 13.9331 8.84801 13.4925 9.05859 12.8818C9.07702 12.8284 9.09258 12.7732 9.11035 12.7178C9.65998 12.6872 10.1964 12.6354 10.7148 12.5615C10.6409 12.8457 10.5642 13.1169 10.4766 13.3711C10.2348 14.0722 9.93027 14.6936 9.55469 15.1543C9.18379 15.6092 8.66258 16 8 16C7.36812 16 6.55348 15.4062 5.85352 13.875C5.23926 12.5311 4.82433 10.6728 4.75977 8.57324C4.75476 8.38471 4.75005 8.19456 4.75 8.00293C5.17751 7.9242 5.68594 7.85929 6.25195 7.81543C6.25136 7.87673 6.25 7.93825 6.25 8Z" fill="#0B2026"/><path d="M12.5615 5.2832C12.8458 5.35734 13.1168 5.43574 13.3711 5.52344C14.0722 5.76523 14.6936 6.06973 15.1543 6.44531C15.6092 6.81621 16 7.33742 16 8C16 8.63188 15.4062 9.44652 13.875 10.1465C12.5942 10.7319 10.846 11.1369 8.86719 11.2295C8.58296 11.2412 8.29465 11.249 8.00293 11.249C7.92423 10.8215 7.85926 10.3131 7.81543 9.74707C7.87673 9.74768 7.93825 9.75 8 9.75C9.41469 9.75 10.7163 9.63565 11.6826 9.44238C12.1596 9.34699 12.5943 9.22505 12.9277 9.06738C12.9856 9.04003 13.0448 9.00826 13.1045 8.97363C13.6001 8.78508 13.9677 8.58593 14.2061 8.3916C14.4849 8.16431 14.5 8.02778 14.5 8C14.5 7.97222 14.4849 7.83569 14.2061 7.6084C13.9331 7.38586 13.4925 7.15199 12.8818 6.94141C12.8283 6.92295 12.7733 6.9065 12.7178 6.88867C12.6871 6.33871 12.6355 5.80196 12.5615 5.2832Z" fill="#0B2026"/><path d="M7.99609 4.75C8.07483 5.17753 8.13973 5.68589 8.18359 6.25195C8.12262 6.25136 8.06142 6.25 8 6.25C6.58531 6.25 5.28371 6.36435 4.31738 6.55762C3.84041 6.65301 3.40571 6.77495 3.07227 6.93262C3.01432 6.96002 2.95437 6.99069 2.89453 7.02539C2.39918 7.21389 2.03224 7.41414 1.79395 7.6084C1.51515 7.83569 1.5 7.97222 1.5 8C1.5 8.02778 1.51515 8.16431 1.79395 8.3916C2.06694 8.61414 2.50746 8.84801 3.11816 9.05859C3.17131 9.07692 3.22616 9.09266 3.28125 9.11035C3.31186 9.66029 3.36353 10.1971 3.4375 10.7158C3.15362 10.6419 2.88289 10.5641 2.62891 10.4766C1.92776 10.2348 1.30642 9.93027 0.845703 9.55469C0.390761 9.18379 2.33127e-07 8.66258 0 8C4.81473e-08 7.36812 0.593829 6.55348 2.125 5.85352C3.46781 5.23974 5.32436 4.82483 7.42188 4.75977C7.61169 4.7547 7.80315 4.75006 7.99609 4.75Z" fill="#0B2026"/><path d="M8 0C8.63188 5.52408e-08 9.44652 0.593829 10.1465 2.125C10.7538 3.45362 11.1666 5.28519 11.2383 7.35547C11.2448 7.56699 11.249 7.78061 11.249 7.99609C10.8214 8.0748 10.3131 8.13976 9.74707 8.18359C9.74768 8.12261 9.75 8.06143 9.75 8C9.75 6.58531 9.63565 5.28371 9.44238 4.31738C9.34699 3.84041 9.22505 3.40571 9.06738 3.07227C9.03992 3.01418 9.00844 2.95451 8.97363 2.89453C8.78517 2.39935 8.58581 2.03219 8.3916 1.79395C8.16431 1.51515 8.02777 1.5 8 1.5C7.97222 1.5 7.83569 1.51515 7.6084 1.79395C7.38586 2.06694 7.15199 2.50746 6.94141 3.11816C6.92305 3.1714 6.90641 3.22607 6.88867 3.28125C6.33874 3.31189 5.80193 3.3635 5.2832 3.4375C5.35731 3.15357 5.43584 2.88295 5.52344 2.62891C5.76523 1.92776 6.06973 1.30642 6.44531 0.845703C6.81621 0.390761 7.33742 9.37695e-08 8 0Z" fill="#0B2026"/></svg> has been enabled. </p>
# MAGIC     <div class="code-card">
# MAGIC       <div class="code-card-header">
# MAGIC         <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt</span>
# MAGIC         <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC           <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC           <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC           <span class="copy-label">Copy</span>
# MAGIC         </button>
# MAGIC       </div>
# MAGIC       <div class="code-card-body">What files do you have access to?</div>
# MAGIC     </div>
# MAGIC     <p>The image to the right is an example output &mdash; your response may look slightly different. Again, we have not explicitly indexed this PDF at this point. Also note the source information returned. It will link you back to the volume where the information lives.</p>
# MAGIC   </div>
# MAGIC   <div style="flex: 1; min-width: 280px; text-align: center; padding: 8px 8px 16px 8px;">
# MAGIC     <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/genie-1.png" alt="genie-1.png" title="genie-1.png" style="max-width: 75%; border: 1px solid #DCE0E2; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);" />
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC <style>
# MAGIC   .code-card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #DCE0E2; border-radius: 0.5rem; overflow: hidden; background: #F9F7F4; max-width: 100%; margin: 0.5rem 0; }
# MAGIC   .code-card-header { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.75rem; background: #EEEDE9; border-bottom: 1px solid #DCE0E2; }
# MAGIC   .code-card-lang { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #5A6F77; }
# MAGIC   .code-card-copy { display: inline-flex; align-items: center; gap: 0.3rem; background: none; border: none; color: #5A6F77; cursor: pointer; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500; transition: color 0.15s, background 0.15s; }
# MAGIC   .code-card-copy:hover { color: #1B3139; background: rgba(0,0,0,0.05); }
# MAGIC   .code-card-copy.copied { color: #2272B4; }
# MAGIC   .code-card-copy svg { width: 14px; height: 14px; }
# MAGIC   .code-card-body { margin: 0; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.82rem; line-height: 1.55; color: #1B3139; white-space: pre-wrap; word-break: break-word; }
# MAGIC </style>

# COMMAND ----------

# DBTITLE 1,Cell 11
# MAGIC %md-sandbox
# MAGIC ## D. Add Another Volume as a Document Source
# MAGIC As mentioned before, we also have the ability to perform global indexing on a volume via a feature called **Content Search**. ([AWS](https://docs.databricks.com/aws/en/genie-agents/volumes) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/genie-agents/volumes) | [GCP](https://docs.databricks.com/gcp/en/genie-agents/volumes))
# MAGIC
# MAGIC Your lab environment has been configured so these features have been turned on for you, but you can read the list of requirements [here](https://docs.databricks.com/aws/en/volumes/content-search#requirements). 
# MAGIC
# MAGIC 1. In your Genie Agent's configuration menu, select **Sources** and click **Add**. 
# MAGIC 1. Select **Volume**. 
# MAGIC 1. Select the other volume `listings_vol` and click **Confirm**.
# MAGIC     - Like before, you will see the volume indicating that it's **Indexing**. Wait for it to finish Indexing before continuing to the next sesion. 
# MAGIC
# MAGIC In the meantime, while your volume is being indexed, head on over to the [official Databricks documentation](https://docs.databricks.com/aws/en/volumes/content-search) to read about current limitations, billing, etc.

# COMMAND ----------

# DBTITLE 1,D4 - Inspect the Indexed Tables - ADMIN ACCESS NEEDED
# MAGIC %md-sandbox
# MAGIC ### D4. Inspect the Indexed Tables - ADMIN ACCESS NEEDED
# MAGIC
# MAGIC <div style="display: flex; align-items: stretch; gap: 24px; flex-wrap: wrap; padding: 0 10px 40px 0;">
# MAGIC   <div style="flex: 1; min-width: 280px; display: flex; flex-direction: column;">
# MAGIC     <p>Once your volume has been indexed (you will see a message that indicates <strong>Content search</strong> is <strong>Ready</strong>), you can inspect what was just performed for you.</p>
# MAGIC     <ol>
# MAGIC       <li>In the <strong>Catalog Explorer</strong> <svg width="18" height="18" viewBox="0 0 16 16" fill="none" style="vertical-align: -3px; display: inline-block;"><path fill-rule="evenodd" clip-rule="evenodd" d="M8.64576 0.368453C8.51082 0.140099 8.2653 0 8.00006 0C7.73482 0 7.4893 0.140099 7.35437 0.368453L4.10437 5.86845C3.96736 6.1003 3.96518 6.38781 4.09865 6.62171C4.23213 6.85561 4.48076 7 4.75006 7H11.2501C11.5194 7 11.768 6.85561 11.9015 6.62171C12.0349 6.38781 12.0328 6.1003 11.8958 5.86845L8.64576 0.368453ZM8.00006 2.22426L9.93572 5.5H6.0644L8.00006 2.22426ZM8.5 9.25C8.5 8.83579 8.83579 8.5 9.25 8.5H14.25C14.6642 8.5 15 8.83579 15 9.25V14.25C15 14.6642 14.6642 15 14.25 15H9.25C8.83579 15 8.5 14.6642 8.5 14.25V9.25ZM10 10V13.5H13.5V10H10ZM4.25 10C3.2835 10 2.5 10.7835 2.5 11.75C2.5 12.7165 3.2835 13.5 4.25 13.5C5.2165 13.5 6 12.7165 6 11.75C6 10.7835 5.2165 10 4.25 10ZM1 11.75C1 9.95507 2.45507 8.5 4.25 8.5C6.04493 8.5 7.5 9.95507 7.5 11.75C7.5 13.5449 6.04493 15 4.25 15C2.45507 15 1 13.5449 1 11.75Z" fill="#0B2026"/></svg>, click on <strong>Details</strong>.
# MAGIC         <ul>
# MAGIC           <li>Here you will find metadata about the volume that has been indexed as part of the previous step (`listings_vol`) (when it was created, who created it, where it's physically stored in cloud storage, etc).</li>
# MAGIC           <li>You will also see under <strong>Content search</strong> the synced status, the last indexed timestamp, how many files were processed during the last sync, and the total number of files.</li>
# MAGIC         </ul>
# MAGIC       </li>
# MAGIC       <li>Navigate to the waffle at the top right next to your profile icon click on it and right click on <strong>Lakebase Postgres</strong> and select <strong>Open Link in New Tab</strong>.</li>
# MAGIC       <li>Here you will find a Lakebase project with a name like <code>vi-ws-XXX-lakebase</code>. This is the lakebase project created by a volume index system service principal.</li>
# MAGIC       <li>Click on the Lakebase project identified in the previous step and click on <strong>Overview</strong>.</li>
# MAGIC       <li>You will need to give yourself permission to view the tables since technically the volume index SSP owns this project. Click on <strong>Roles &amp; Databases</strong>. Click <strong>Add role</strong> and add yourself as a <code>databricks_superuser</code>. Note you will have to search for your labuser name (again, see the table output from running the classroom setup script up above).</li>
# MAGIC       <li>Click on <strong>Tables</strong> and locate the table that says something like <code>index_vi_XXX</code> and click on it. Here you will see the schema shown to the right. <strong>Note:</strong> You do not have access to view or query the table data unless you are a Databricks admin — if you have not completed the previous step or have not yet added yourself as a <code>databricks_superuser</code>, you will see the permission denied error shown below the schema.</li>
# MAGIC     </ol>
# MAGIC     <div style="margin-top: auto; text-align: center; padding-top: 16px;">
# MAGIC       <img src="https://files.training.databricks.com/binder/prod_main/building-rag-agents-with-agent-bricks-en_us-2.0.0/images/20261014T152401Z/Building RAG Agents with Agent Bricks/Includes/images/permission-denied.png" alt="Permission denied error" style="max-width: 100%; border: 1px solid #DCE0E2; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);" />
# MAGIC       <p style="font-size: 13px; color: #5A6F77; margin-top: 10px; font-style: italic;">Without admin access, you will not have the ability to give yourself permission to view the tables</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div style="flex: 0 1 380px; min-width: 280px; padding-top: 8px;">
# MAGIC     <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
# MAGIC       <div style="background:#FFFFFF; border:1.5px solid #1B5162; border-radius:14px; overflow:visible; box-shadow: 0 2px 8px rgba(0,0,0,0.12);">
# MAGIC         <div style="border-bottom:1.5px solid #E2D9D5; padding:11px 18px;">
# MAGIC           <span style="background:rgba(27,81,98,0.10); padding:2px 10px; border-radius:6px; font-size:11px; font-weight:700; color:#1B5162; letter-spacing:0.5px;">LAKEBASE</span>
# MAGIC           <span style="font-size:14px; font-weight:700; color:#0b2026; margin-left:8px;">Index Table Schema</span>
# MAGIC         </div>
# MAGIC         <div style="padding:12px 18px;">
# MAGIC           <table style="width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.7;">
# MAGIC             <tr><td colspan="3" style="padding:8px 0 3px; font-size:10px; font-weight:700; color:#5A6F77; letter-spacing:0.8px; text-transform:uppercase; border:none;">Document Metadata</td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">filename</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">modificationTime</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TIMESTAMP WITH TIME ZONE</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">doc_uri</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">doc_type</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td colspan="3" style="padding:10px 0 3px; font-size:10px; font-weight:700; color:#5A6F77; letter-spacing:0.8px; text-transform:uppercase; border:none;">Parsing Config</td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">parser</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">splitter</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">chunk_size</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">INTEGER</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">chunk_overlap</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">INTEGER</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td colspan="3" style="padding:10px 0 3px; font-size:10px; font-weight:700; color:#5A6F77; letter-spacing:0.8px; text-transform:uppercase; border:none;">Chunks</td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">chunk_id</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#1976d2; font-weight:700; font-size:11px; padding:1px 8px; border:none;">PRIMARY KEY</td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">chunk_pos_id</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">INTEGER</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">chunk_text</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">metadata</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#0b2026; padding:1px 0; border:none;">parsed_doc_summary</td><td style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#D32F2F; padding:1px 8px; border:none;">TEXT</td><td style="border:none;"></td></tr>
# MAGIC             <tr><td colspan="3" style="padding:10px 0 3px; font-size:10px; font-weight:700; color:#FF5F46; letter-spacing:0.8px; text-transform:uppercase; border:none;">Search Columns</td></tr>
# MAGIC             <tr><td colspan="3" style="padding:0; border:none;"><div style="background:#FFF3F0; border:1px solid #FFCDD2; border-radius:6px; padding:8px 10px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:11.5px; line-height:1.7;">
# MAGIC               <div>embedding_model <span style="color:#D32F2F;">TEXT</span></div>
# MAGIC               <div>__db_chunk_text_vector <span style="color:#D32F2F;">JSONB</span></div>
# MAGIC               <div>__db_chunk_text_pgvector <span style="color:#D32F2F;">VECTOR(1024)</span></div>
# MAGIC               <div style="color:#618794; font-size:10px; padding-left:14px;">GENERATED ALWAYS AS ... STORED</div>
# MAGIC               <div>__db_chunk_text_tsvector <span style="color:#D32F2F;">TSVECTOR</span></div>
# MAGIC               <div style="color:#618794; font-size:10px; padding-left:14px;">GENERATED ALWAYS AS (to_tsvector(...))</div>
# MAGIC             </div></td></tr>
# MAGIC             <tr><td colspan="3" style="padding:10px 0 3px; font-size:10px; font-weight:700; color:#00A972; letter-spacing:0.8px; text-transform:uppercase; border:none;">Indexes</td></tr>
# MAGIC             <tr><td colspan="3" style="padding:0; border:none;"><div style="background:#E6F5EE; border:1px solid #A5D6C0; border-radius:6px; padding:8px 10px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:11.5px; line-height:1.7;">
# MAGIC               <div><span style="color:#00A972; font-weight:700;">LAKEBASE_ANN</span> (__db_chunk_text_pgvector)</div>
# MAGIC               <div><span style="color:#00A972; font-weight:700;">LAKEBASE_BM25</span> (__db_chunk_text_tsvector)</div>
# MAGIC               <div><span style="color:#00A972; font-weight:700;">BTREE</span> (chunk_id)</div>
# MAGIC             </div></td></tr>
# MAGIC           </table>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #f44336; background: rgba(244,67,54,0.12); padding: 14px 18px; border-radius: 4px; margin: 16px 0; font-family: sans-serif;">
# MAGIC   <strong style="display:block; color:#c62828; margin-bottom:6px; font-size:15pt;">Warning</strong>
# MAGIC   <div style="color:#333; font-size:14pt; line-height:1.55;">
# MAGIC
# MAGIC This can take several minutes to complete. You should see **Ready** next to your volume in the **Genie Agent** UI. Once ready, proceed to ask your **Genie Agent** questions.
# MAGIC
# MAGIC   </div>
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## E. Query the Genie Agent in Agent Mode
# MAGIC
# MAGIC Now ask questions about your documents.
# MAGIC
# MAGIC 1. Open the chat box in your Genie Agent and turn on **Agent mode**. You must use Agent mode to answer questions about files.
# MAGIC
# MAGIC 2. Try the three prompts below. The first targets the **house rules** PDF, the second targets the **listings** PDF, and the third is a **hybrid** that requires both documents.
# MAGIC
# MAGIC 3. For each answer, expand the **citations** to confirm the response is grounded in the source PDF.
# MAGIC
# MAGIC <div class="code-card">
# MAGIC   <div class="code-card-header">
# MAGIC     <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt — House Rules PDF</span>
# MAGIC     <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body">What is the pet policy, and what are the quiet hours guests must follow?</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="code-card">
# MAGIC   <div class="code-card-header">
# MAGIC     <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt — Listings PDF</span>
# MAGIC     <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body">Which listings are near Golden Gate Park, and what amenities do they offer?</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="code-card">
# MAGIC   <div class="code-card-header">
# MAGIC     <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt — Hybrid (Both PDFs)</span>
# MAGIC     <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body">For a listing near Golden Gate Park, summarize its amenities and the house rules a guest would need to follow.</div>
# MAGIC </div>
# MAGIC
# MAGIC <style>
# MAGIC   .code-card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #DCE0E2; border-radius: 0.5rem; overflow: hidden; background: #F9F7F4; max-width: 100%; margin: 0.5rem 0; }
# MAGIC   .code-card-header { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.75rem; background: #EEEDE9; border-bottom: 1px solid #DCE0E2; }
# MAGIC   .code-card-lang { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #5A6F77; }
# MAGIC   .code-card-copy { display: inline-flex; align-items: center; gap: 0.3rem; background: none; border: none; color: #5A6F77; cursor: pointer; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500; transition: color 0.15s, background 0.15s; }
# MAGIC   .code-card-copy:hover { color: #1B3139; background: rgba(0,0,0,0.05); }
# MAGIC   .code-card-copy.copied { color: #2272B4; }
# MAGIC   .code-card-copy svg { width: 14px; height: 14px; }
# MAGIC   .code-card-body { margin: 0; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.82rem; line-height: 1.55; color: #1B3139; white-space: pre-wrap; word-break: break-word; }
# MAGIC </style>

# COMMAND ----------

# MAGIC %md
# MAGIC ## F. Adding New Documents & Syncing
# MAGIC When you load a new document to your volume, the parse/index pipeline is not automatic.  
# MAGIC
# MAGIC To refresh content search so that it reflects your changes, click **Sync now** in the **Sources** section in the **Genie Agent**. In this section, you'll experience this firsthand by adding a new document and then syncing the index.
# MAGIC
# MAGIC Note that in general, you need `CAN MANAGE` permissions on your volume to trigger the sync.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### F1. Ask About an Invoice (Before Indexing)
# MAGIC
# MAGIC Open your Genie Agent in **Agent mode** and ask about an invoice. Since no invoice has been added to the indexed volume yet, the agent should not be able to find any relevant information.
# MAGIC
# MAGIC <div class="code-card">
# MAGIC   <div class="code-card-header">
# MAGIC     <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt — Invoice Query</span>
# MAGIC     <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body">What is the total amount due on the invoice, and who is it from?</div>
# MAGIC </div>
# MAGIC
# MAGIC <style>
# MAGIC   .code-card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #DCE0E2; border-radius: 0.5rem; overflow: hidden; background: #F9F7F4; max-width: 100%; margin: 0.5rem 0; }
# MAGIC   .code-card-header { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.75rem; background: #EEEDE9; border-bottom: 1px solid #DCE0E2; }
# MAGIC   .code-card-lang { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #5A6F77; }
# MAGIC   .code-card-copy { display: inline-flex; align-items: center; gap: 0.3rem; background: none; border: none; color: #5A6F77; cursor: pointer; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500; transition: color 0.15s, background 0.15s; }
# MAGIC   .code-card-copy:hover { color: #1B3139; background: rgba(0,0,0,0.05); }
# MAGIC   .code-card-copy.copied { color: #2272B4; }
# MAGIC   .code-card-copy svg { width: 14px; height: 14px; }
# MAGIC   .code-card-body { margin: 0; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.82rem; line-height: 1.55; color: #1B3139; white-space: pre-wrap; word-break: break-word; }
# MAGIC </style>

# COMMAND ----------

# DBTITLE 1,F2 Copy Invoice Markdown
# MAGIC %md
# MAGIC ### F2. Copy an Invoice PDF into the Volume
# MAGIC
# MAGIC The Genie agent couldn't answer because no invoice exists in the indexed volume yet. Let's fix that by copying an invoice PDF into the `listings_vol` volume under a new `invoices/` subdirectory.
# MAGIC
# MAGIC Run the cell below to copy `invoice_01.pdf` from the course materials into your volume.

# COMMAND ----------

# DBTITLE 1,F2 Copy Invoice Code
import shutil, os

# Source file (in the workspace, relative to the notebook directory)
source_file = "./Includes/data/invoices/invoice_01.pdf"

# Destination: new "invoices" subdirectory inside the listings volume
dest_dir = f"/Volumes/{catalog_name}/{schema_name}/listings_vol/invoices"
os.makedirs(dest_dir, exist_ok=True)

# Copy the invoice PDF into the volume
dest_file = os.path.join(dest_dir, "invoice_01.pdf")
shutil.copy(source_file, dest_file)
print(f"Copied {source_file} → {dest_file}")

# COMMAND ----------

# DBTITLE 1,F3 Sync Volume Markdown
# MAGIC %md
# MAGIC ### F3. Sync the Volume
# MAGIC
# MAGIC The file has been copied, but content search hasn't picked it up yet — remember, indexing is **not automatic**. You need to manually trigger a sync:
# MAGIC
# MAGIC 1. In a new tab navigate to **Catalog Explorer** and open the `listings_vol` volume.
# MAGIC
# MAGIC 2. Select **Details**
# MAGIC
# MAGIC 3. In the **Content search** section, click **Sync now**.
# MAGIC
# MAGIC 4. Wait for the sync to complete — the status should change to **Synced** (this may take a few minutes).
# MAGIC
# MAGIC Once the status shows **Synced**, continue to the next section.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #f44336; background: rgba(244,67,54,0.12); padding: 14px 18px; border-radius: 4px; margin: 16px 0; font-family: sans-serif;">
# MAGIC   <strong style="display:block; color:#c62828; margin-bottom:6px; font-size:15pt;">Warning</strong>
# MAGIC   <div style="color:#333; font-size:14pt; line-height:1.55;">
# MAGIC
# MAGIC This can take several minutes to complete. You should see **Ready** on your volume in **Catalog Explorer**. Once ready, proceed to ask your **Genie Agent** questions.
# MAGIC
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,F4 Re-query Invoice Markdown
# MAGIC %md-sandbox
# MAGIC ### F4. Ask About the Invoice Again (After Indexing)
# MAGIC
# MAGIC Now that the invoice has been indexed, ask the same question again in your Genie Agent. This time, the agent should be able to find and cite the invoice.
# MAGIC
# MAGIC <div class="code-card">
# MAGIC   <div class="code-card-header">
# MAGIC     <span class="code-card-lang"><svg width="26" height="26" viewBox="-1.108 -1.458 202.915 202.915" fill="none" style="vertical-align:-4px; margin-right:5px; display:inline-block;"><path d="M179.614 80.7871C167.525 100.208 149.376 126.933 141.272 133.79C133.167 140.647 121.149 153.009 93.2907 153.009C73.7741 153.009 56.9864 140.474 49.5352 122.498C49.5353 122.519 49.5356 122.54 49.5358 122.561C48.4799 119.938 45.9132 118.085 42.9121 118.085C38.969 118.085 35.7721 121.282 35.7721 125.225C35.7722 126.15 35.9484 126.997 36.2689 127.752C37.984 131.793 42.5469 131.965 42.5469 144.569C42.5469 143.103 42.5469 146.404 42.5469 144.569C30.6947 144.569 21.0865 134.96 21.0863 123.108C21.0863 111.256 30.6945 101.647 42.5469 101.647H45.4249V101.644H79.5153C82.062 101.644 84.5067 102.722 86.0978 104.71C88.8856 108.194 91.0222 111.438 92.5969 114.114C94.056 116.594 98.4255 116.618 99.8699 114.13C101.393 111.506 103.416 108.337 105.98 104.929C107.573 102.813 110.103 101.644 112.752 101.644H125.427C143.951 101.644 154.105 92.7623 159.289 86.3612C161.972 83.049 165.782 80.7871 170.045 80.7871H179.614Z" fill="#FABFBA"/><path d="M96.3843 111.768C96.3842 87.8629 77.3194 68.4739 53.768 68.3843C77.3196 68.2949 96.3845 48.9049 96.3845 25C96.3845 25 96.3845 25 96.3845 25C96.3845 48.9602 115.538 68.3838 139.165 68.3838C115.538 68.3838 96.3844 87.8075 96.3843 111.768Z" fill="#FF5F46"/><path d="M81.7909 159.789C75.0244 159.789 69.5392 166.599 69.5392 175H126.368C126.368 166.599 120.883 159.789 114.116 159.789H81.7909Z" fill="#FF5F46"/></svg>Prompt — Invoice Query (After Sync)</span>
# MAGIC     <button class="code-card-copy" onclick="(function(b){var c=b.closest('.code-card').querySelector('.code-card-body').textContent.trim();var t=document.createElement('textarea');t.value=c;t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();var ok=false;try{ok=document.execCommand('copy')}catch(e){}document.body.removeChild(t);if(ok){var l=b.querySelector('.copy-label'),ci=b.querySelector('.copy-icon'),chi=b.querySelector('.check-icon');ci.style.display='none';chi.style.display='';l.textContent='Copied!';b.classList.add('copied');setTimeout(function(){ci.style.display='';chi.style.display='none';l.textContent='Copy';b.classList.remove('copied')},2000)}})(this)">
# MAGIC       <svg class="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
# MAGIC       <svg class="check-icon" style="display:none" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
# MAGIC       <span class="copy-label">Copy</span>
# MAGIC     </button>
# MAGIC   </div>
# MAGIC   <div class="code-card-body">What is the total amount due on the invoice, and who is it from?</div>
# MAGIC </div>
# MAGIC
# MAGIC Expand the **citations** to confirm the response is grounded in the source PDF.
# MAGIC
# MAGIC <style>
# MAGIC   .code-card { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #DCE0E2; border-radius: 0.5rem; overflow: hidden; background: #F9F7F4; max-width: 100%; margin: 0.5rem 0; }
# MAGIC   .code-card-header { display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.75rem; background: #EEEDE9; border-bottom: 1px solid #DCE0E2; }
# MAGIC   .code-card-lang { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #5A6F77; }
# MAGIC   .code-card-copy { display: inline-flex; align-items: center; gap: 0.3rem; background: none; border: none; color: #5A6F77; cursor: pointer; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 500; transition: color 0.15s, background 0.15s; }
# MAGIC   .code-card-copy:hover { color: #1B3139; background: rgba(0,0,0,0.05); }
# MAGIC   .code-card-copy.copied { color: #2272B4; }
# MAGIC   .code-card-copy svg { width: 14px; height: 14px; }
# MAGIC   .code-card-body { margin: 0; padding: 1rem; overflow-x: auto; font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 0.82rem; line-height: 1.55; color: #1B3139; white-space: pre-wrap; word-break: break-word; }
# MAGIC </style>

# COMMAND ----------

# MAGIC %md
# MAGIC ## G. Next Steps on Improving Your Genie Agent
# MAGIC Tuning a Genie Agent is outside the scope of this course, but you can explore curating table and column metadata, adding SQL expressions, refining metadata information, and more. 
# MAGIC
# MAGIC See the Databricks Academy course [Building Reliable Conversational Agents with Genie](https://www.databricks.com/training/catalog/building-reliable-conversational-agents-with-genie-5222) for further training.
# MAGIC
# MAGIC Documentation: **Tune Genie Agent quality** [AWS](https://docs.databricks.com/aws/en/genie-agents/tune-quality) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/genie-agents/tune-quality) | [GCP](https://docs.databricks.com/gcp/en/genie-agents/tune-quality)

# COMMAND ----------

# MAGIC %md
# MAGIC ## H. Conclusion
# MAGIC In this demo, you:
# MAGIC
# MAGIC 1. **Inspected** a Unity Catalog volume of Airbnb PDFs created during setup
# MAGIC 2. **Created** a Genie Agent and attached the volume as a document source
# MAGIC 3. **Enabled content search** so Genie could index and retrieve from the files
# MAGIC 4. **Queried** the documents in Agent mode and validated cited, grounded responses
# MAGIC
# MAGIC This shows how Genie can serve as a natural-language interface over **unstructured documents**, not just SQL tables.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>