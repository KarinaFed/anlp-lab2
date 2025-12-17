"""Main entry point for the Multi-Agent Study & Productivity Assistant"""
import asyncio
import sys
from src.graph import MultiAgentSystem


async def main():
    """Main function to run the multi-agent system"""
    print("=" * 70)
    print("Multi-Agent Study & Productivity Assistant")
    print("=" * 70)
    print()
    
    # Initialize system
    system = MultiAgentSystem()
    
    # Example queries
    if len(sys.argv) > 1:
        # Query from command line
        query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("Enter your query (or 'quit' to exit):")
        query = input("> ")
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        return
    
    # Process query
    print(f"\nProcessing query: {query}\n")
    result = await system.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    
    if result.get("final_response"):
        response = result["final_response"]
        print(f"\n{response.answer}\n")
        print(f"Agents involved: {', '.join(response.agents_involved)}")
        print(f"Tools used: {', '.join(response.tools_used) if response.tools_used else 'None'}")
        print(f"Memory accessed: {response.memory_accessed}")
        print(f"Confidence: {response.confidence}")
    else:
        print("\nNo response generated.")
        if result.get("error"):
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

