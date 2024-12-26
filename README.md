# llm-equities-research
Backend for the VueJS frontend (seperate repo).

### Repo background
**/dba**
- Postgres database manager
- Data models

**/src**
- Engines (engine: a tool used to "build" an output for GP, e.g. a financial model or a one pager)
- Wizard (the wizard contains the LLM wrappers/management module)
- Other API wrappers
- Configs
- Various utils

**/app**
- Flask app
- Endpoint decorators

**/static**
- Local stage for the Flask API when handling files

### Standards
- Use a context manager for db session usage
````
with db_session() as session:
    # execute session code here
````

### Future State Code Improvements Suggestions (ranked in order of severity, top to bottom)
- [ ] Rebuild Cloud Function logic for updated OAI API usage
- [ ] Merge endpoints, add more logic to minimize volume
- [ ] Migrate core Flask logic to seperate modules
- [ ] Add more specific error/log try/excepting (possibly with custom errors)
- [ ] User-based api access, JWT auth
