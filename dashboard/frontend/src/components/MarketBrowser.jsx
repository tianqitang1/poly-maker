import React, { useState, useEffect } from 'react';

export const MarketBrowser = ({ onAddMarket }) => {
    const [markets, setMarkets] = useState([]);
    const [cursor, setCursor] = useState("");
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");

    useEffect(() => {
        fetchMarkets();
    }, []);

    const fetchMarkets = async (nextCursor = "") => {
        setLoading(true);
        try {
            const res = await fetch(`http://localhost:8001/markets?limit=50&cursor=${nextCursor}`);
            const data = await res.json();
            if (nextCursor) {
                setMarkets(prev => [...prev, ...data.markets]);
            } else {
                setMarkets(data.markets);
            }
            setCursor(data.next_cursor);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const filteredMarkets = markets.filter(m => 
        m.question.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="card">
            <div className="flex-between">
                <h2>Browse Active Markets</h2>
                <input 
                    type="text" 
                    placeholder="Search markets..." 
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{maxWidth: '300px'}}
                />
            </div>
            
            <div className="grid-2">
                {filteredMarkets.map(m => (
                    <div key={m.condition_id} className="market-card" style={{border: '1px solid #333', padding: '15px', borderRadius: '8px'}}>
                        <h4>{m.question}</h4>
                        <p style={{fontSize: '0.8rem', color: '#aaa'}}>End: {m.end_date}</p>
                        <button className="btn" onClick={() => onAddMarket(m)}>Add to Selection</button>
                    </div>
                ))}
            </div>
            
            {cursor && (
                <div style={{textAlign: 'center', marginTop: '20px'}}>
                    <button className="btn" onClick={() => fetchMarkets(cursor)} disabled={loading}>
                        {loading ? "Loading..." : "Load More"}
                    </button>
                </div>
            )}
        </div>
    );
};
